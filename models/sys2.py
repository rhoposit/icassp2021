# ==================================================================================================
# Copyright (c) 2021, Jennifer Williams and Yamagishi Laboratory, National Institute of Informatics
# Author: Jennifer Williams (j.williams@ed.ac.uk)
# All rights reserved.
# ==================================================================================================


import math, pickle, os
import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from utils.dsp import *
import sys
import time
from layers.overtone import Overtone
from layers.vector_quant import VectorQuant
from layers.downsampling_encoder import DownsamplingEncoder
import utils.env as env
import utils.logger as logger
import random
import pytorch_warmup as warmup

class Model(nn.Module) :
    def __init__(self, rnn_dims, fc_dims, global_decoder_cond_dims, upsample_factors, normalize_vq=False,noise_x=False, noise_y=False):
        super().__init__()
        self.n_classes = 256
        self.overtone = Overtone(rnn_dims, fc_dims, 128, global_decoder_cond_dims)
        self.vq = VectorQuant(1, 512, 128, normalize=normalize_vq)
        self.noise_x = noise_x
        self.noise_y = noise_y
        encoder_layers = [
            (2, 4, 1),
            (2, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            ]
        self.encoder = DownsamplingEncoder(128, encoder_layers)
        self.frame_advantage = 15
        self.num_params()
        self.left = self.pad_left()
        self.win = 16 * self.total_scale()
        self.right = self.pad_right()


        

    def forward(self, global_decoder_cond, x, samples):
        continuous = self.encoder(samples)
        discrete, vq_pen, encoder_pen, entropy, phn_codes = self.vq(continuous.unsqueeze(2))
        return self.overtone(x, discrete.squeeze(2), global_decoder_cond), vq_pen.mean(), encoder_pen.mean(), entropy

    def after_update(self):
        self.overtone.after_update()
        self.vq.after_update()

    def generate(self, samples, global_decoder_cond, deterministic=False, use_half=False, verbose=False):
        self.eval()
        with torch.no_grad() :
            continuous = self.encoder(samples)
            discrete, vq_pen, encoder_pen, entropy = self.vq(continuous.unsqueeze(2))
            code_x = discrete.squeeze(2)
            output = self.overtone.generate(code_x, global_decoder_cond, use_half=use_half, verbose=verbose)
        return output


    def forward_validate(self, global_decoder_cond, x, samples):
        self.eval()
        with torch.no_grad() :
            continuous = self.encoder(samples)
            discrete, vq_pen, encoder_pen, entropy, phn_codes = self.vq(continuous.unsqueeze(2))
        self.train()
        return self.overtone(x, discrete.squeeze(2), global_decoder_cond), vq_pen.mean(), encoder_pen.mean()


    
    def num_params(self) :
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        logger.log('Trainable Parameters: %.3f million' % parameters)

    def load_state_dict(self, dict, strict=True):
        if strict:
            return super().load_state_dict(self.upgrade_state_dict(dict))
        else:
            my_dict = self.state_dict()
            new_dict = {}
            for key, val in dict.items():
                if key not in my_dict:
                    logger.log(f'Ignoring {key} because no such parameter exists')
                elif val.size() != my_dict[key].size():
                    logger.log(f'Ignoring {key} because of size mismatch')
                else:
                    logger.log(f'Loading {key}')
                    new_dict[key] = val
            return super().load_state_dict(new_dict, strict=False)

    def upgrade_state_dict(self, state_dict):
        out_dict = state_dict.copy()
        return out_dict

    def freeze_encoder(self):
        for name, param in self.named_parameters():
            if name.startswith('encoder.') or name.startswith('vq.'):
                logger.log(f'Freezing {name}')
                param.requires_grad = False
            else:
                logger.log(f'Not freezing {name}')

    def pad_left(self):
        return max(self.pad_left_decoder(), self.pad_left_encoder())

    def pad_left_decoder(self):
        return self.overtone.pad()

    def pad_left_encoder(self):
        return self.encoder.pad_left + (self.overtone.cond_pad - self.frame_advantage) * self.encoder.total_scale

    def pad_right(self):
        return self.frame_advantage * self.encoder.total_scale

    def total_scale(self):
        return self.encoder.total_scale

    def tmp_func(self, batch):
        return env.collate_multispeaker_samples(self.left, self.win, self.right, batch)

    def tmp_func2(self, batch):
        return env.collate_multispeaker_samples_forward(self.left, self.win, self.right, batch)
    
    def do_train(self, paths, dataset, optimiser, epochs, batch_size, num_workers, step, train_sampler, device, lr=1e-4,  spk_lr=0.01, valid_index=[], use_half=False, do_clip=False):

        if use_half:
            import apex
            optimiser = apex.fp16_utils.FP16_Optimizer(optimiser, dynamic_loss_scale=True)
        for p in optimiser.param_groups : p['lr'] = lr
        criterion = nn.NLLLoss().cuda()
        k = 0
        saved_k = 0
        pad_left = self.pad_left()
        pad_left_encoder = self.pad_left_encoder()
        pad_left_decoder = self.pad_left_decoder()
        if self.noise_x:
            extra_pad_right = 127
        else:
            extra_pad_right = 0
        pad_right = self.pad_right() + extra_pad_right
        window = 16 * self.total_scale()
        logger.log(f'pad_left={pad_left_encoder}|{pad_left_decoder}, pad_right={pad_right}, total_scale={self.total_scale()}')

        # from haoyu: slow start for the first 10 epochs
        lr_lambda = lambda epoch: min((epoch) / 10 , 1)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=lr_lambda)
        ## warmup is useful !!!
        # warmup_scheduler = warmup.UntunedLinearWarmup(optimiser)
        # warmup_scheduler.last_step = -1

        for e in range(epochs) :
            self.left = self.pad_left()
            self.win = 16 * self.total_scale()
            self.right = pad_right

            trn_loader = DataLoader(dataset, collate_fn=self.tmp_func, batch_size=batch_size,
                num_workers=num_workers, shuffle=(train_sampler is None),  sampler=train_sampler, pin_memory=True)

            start = time.time()
            running_loss_c = 0.
            running_loss_f = 0.
            running_loss_vq = 0.
            running_loss_vqc = 0.
            running_entropy = 0.
            running_max_grad = 0.
            running_max_grad_name = ""

            iters = len(trn_loader)

            for i, (speaker, wave16) in enumerate(trn_loader) :

                speaker = speaker.cuda(device)
                wave16 = wave16.cuda(device)

                coarse = (wave16 + 2**15) // 256
                fine = (wave16 + 2**15) % 256

                coarse_f = coarse.float() / 127.5 - 1.
                fine_f = fine.float() / 127.5 - 1.
                total_f = (wave16.float() + 0.5) / 32767.5

                if self.noise_y:
                    noisy_f = total_f * (0.02 * torch.randn(total_f.size(0), 1).cuda(device)).exp() + 0.003 * torch.randn_like(total_f)
                else:
                    noisy_f = total_f

                if use_half:
                    coarse_f = coarse_f.half()
                    fine_f = fine_f.half()
                    noisy_f = noisy_f.half()

                x = torch.cat([
                    coarse_f[:, pad_left-pad_left_decoder:-pad_right].unsqueeze(-1),
                    fine_f[:, pad_left-pad_left_decoder:-pad_right].unsqueeze(-1),
                    coarse_f[:, pad_left-pad_left_decoder+1:1-pad_right].unsqueeze(-1),
                    ], dim=2)
                y_coarse = coarse[:, pad_left+1:1-pad_right]
                y_fine = fine[:, pad_left+1:1-pad_right]

                if self.noise_x:
                    # Randomly translate the input to the encoder to encourage
                    # translational invariance
                    total_len = coarse_f.size(1)
                    translated = []
                    for j in range(coarse_f.size(0)):
                        shift = random.randrange(256) - 128
                        translated.append(noisy_f[j, pad_left-pad_left_encoder+shift:total_len-extra_pad_right+shift])
                    translated = torch.stack(translated, dim=0)
                else:
                    translated = noisy_f[:, pad_left-pad_left_encoder:]
                p_cf, vq_pen, encoder_pen, entropy = self(speaker, x, translated)
                p_c, p_f = p_cf
                loss_c = criterion(p_c.transpose(1, 2).float(), y_coarse)
                loss_f = criterion(p_f.transpose(1, 2).float(), y_fine)
                encoder_weight = 0.01 * min(1, max(0.1, step / 1000 - 1))
                loss = loss_c + loss_f + vq_pen + encoder_weight * encoder_pen

                optimiser.zero_grad()
                if use_half:
                    optimiser.backward(loss)
                    if do_clip:
                        raise RuntimeError("clipping in half precision is not implemented yet")
                else:
                    loss.backward()
                    if do_clip:
                        max_grad = 0
                        max_grad_name = ""
                        for name, param in self.named_parameters():
                            if param.grad is not None:
                                param_max_grad = param.grad.data.abs().max()
                                if param_max_grad > max_grad:
                                    max_grad = param_max_grad
                                    max_grad_name = name
                                if 1000000 < param_max_grad:
                                    logger.log(f'Very large gradient at {name}: {param_max_grad}')
                        if 100 < max_grad:
                            for param in self.parameters():
                                if param.grad is not None:
                                    if 1000000 < max_grad:
                                        param.grad.data.zero_()
                                    else:
                                        param.grad.data.mul_(100 / max_grad)
                        if running_max_grad < max_grad:
                            running_max_grad = max_grad
                            running_max_grad_name = max_grad_name

                        if 100000 < max_grad:
                            torch.save(self.state_dict(), "bad_model.pyt")
                            raise RuntimeError("Aborting due to crazy gradient (model saved to bad_model.pyt)")
                optimiser.step()
                if e==0 and i==0:
                     lr_scheduler.step()
                     print("schedulre!")
                # lr_scheduler.step()
                # warmup_scheduler.dampen()
                running_loss_c += loss_c.item()
                running_loss_f += loss_f.item()
                running_loss_vq += vq_pen.item()
                running_loss_vqc += encoder_pen.item()
                running_entropy += entropy

                self.after_update()

                speed = (i + 1) / (time.time() - start)
                avg_loss_c = running_loss_c / (i + 1)
                avg_loss_f = running_loss_f / (i + 1)
                avg_loss_vq = running_loss_vq / (i + 1)
                avg_loss_vqc = running_loss_vqc / (i + 1)
                avg_entropy = running_entropy / (i + 1)

                step += 1
                k = step // 1000
                logger.status(f'Epoch:{e+1}/{epochs} -- Bt:{i+1}/{iters}'
f' -- Lc={avg_loss_c:#.4} -- Lf={avg_loss_f:#.4} -- Lvq={avg_loss_vq:#.4} -- Lvqc={avg_loss_vqc:#.4}'
f' -- E:{avg_entropy:#.4}'
f' -- G:{running_max_grad:#.1} {running_max_grad_name}-- S:{speed:#.4} steps/sec -- Step: {k}k\n\n')
            os.makedirs(paths.checkpoint_dir, exist_ok=True)
            torch.save(self.state_dict(), paths.model_path())
            np.save(paths.step_path(), step)
            logger.log_current_status()
            logger.log(f' <saved>; w[0][0] = {self.overtone.wavernn.gru.weight_ih_l0[0][0]}')
            if k > saved_k + 5:
                torch.save(self.state_dict(), paths.model_hist_path(step))
                saved_k = k
                val_loss_c, val_loss_f,val_loss_vq, val_loss_vqc = self.validate2(paths, step, dataset.path, valid_index, device)
                logger.status(f'Valid: -- Lc={val_loss_c:#.4} -- Lf={val_loss_f:#.4} -- Lvq={val_loss_vq:#.4} -- Lvqc={val_loss_vqc:#.4}')



                
    def validate(self, paths, step, data_path, test_index, device, deterministic=False, use_half=False, verbose=False):
        k = step // 1000
        os.makedirs(paths.gen_path(), exist_ok=True)
        all_files = [(i, name) for (i, speaker) in enumerate(test_index) for name in speaker]
        batch_size = 2
        num_batches, leftovers = divmod(len(all_files), batch_size)
        num_batches = 1
        print("Generating: ", num_batches*batch_size)
        for i in range(0, num_batches):
            start = i*batch_size
            end = (i+1)*batch_size+10
            # get a different speaker
            batch = all_files[start:end:10]
            speakers_onehot = [(np.arange(109) == speaker_id).astype(np.long) for speaker_id, name in batch]
            audio_files = [np.load(f'{data_path}/{speaker_id}/{name}.npy') for speaker_id, name in batch]
            n_points = len(audio_files)
            gt = [(x.astype(np.float32) + 0.5) / (2**15 - 0.5) for x in audio_files]
            extended = [np.concatenate([np.zeros(self.pad_left_encoder(), dtype=np.float32), x, np.zeros(self.pad_right(), dtype=np.float32)]) for x in gt]
            speakers = [torch.FloatTensor(speaker.astype(np.float32)) for speaker in speakers_onehot]
            maxlen = max([len(x) for x in extended])
            aligned = [torch.cat([torch.FloatTensor(x).cuda(), torch.zeros(maxlen-len(x)).cuda()]) for x in extended]            
            A = torch.stack(speakers+ list(reversed(speakers)), dim=0).cuda(device)
            B = torch.stack(aligned + aligned, dim=0).cuda(device)
            out = self.forward_generate(A, B, verbose=verbose, use_half=use_half)
#            logger.log(f'out: {out.size()}')
            for i, x in enumerate(gt) :
                librosa.output.write_wav(f'{paths.gen_path()}/{k}k_steps_{i}_target.wav', x, sr=sample_rate)
                audio = out[i][:len(x)].cpu().numpy()
                librosa.output.write_wav(f'{paths.gen_path()}/{k}k_steps_{i}_generated.wav', audio, sr=sample_rate)
                audio_tr = out[n_points+i][:len(x)].cpu().numpy()
                librosa.output.write_wav(f'{paths.gen_path()}/{k}k_steps_{i}_transferred.wav', audio_tr, sr=sample_rate)



