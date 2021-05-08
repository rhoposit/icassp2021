# icassp2021
This is a Pytorch implementation of Dual-Encoder VQVAE mentioned in [our paper](https://arxiv.org/abs/2010.10727).

We introduce a dual learning space that learns a speaker codebook (global features) alongside the original phone codebook (local). The new framework generalizes well to unseen speakers and the learned representations achieve reasonable performance in downstream tasks such as phone recognition and speaker diarization. 

![Framework of Dual-Encoder VQVAE](https://github.com/rhoposit/icassp2021/blob/main/framework.png)

We reconstructed the speech using both [original VQVAE](https://arxiv.org/abs/1711.00937) and [dual-encoder VQVAE](https://arxiv.org/abs/2010.10727). 

In brief, we have done:

1. Extended the original VQ-VAE with a speaker encoder.
2. Changed the global condition to learned VQVAE speaker code. 
3. Trained a model with multi-speaker English corpus.
4. We provide pre-trained models from various system configurations.

# Authors 
Authors of the paper: Jennifer Williams, Yi Zhao, Erica Cooper, Junichi Yamagishi

For any question related to the paper or the scripts, please contact j.williams[email mark]ed.ac.uk.

# Samples
Please find our samples [here](https://rhoposit.github.io/icassp2021/index.html).

# Requirements

Please install the environment in project.yml before using the scripts.
```
conda env create -f project.yml
conda activate project
```


# Preprocessing
1. we recommend that you first trim the leading/trailing silence from the audio
2. normalize the db levels
3. use the provided pre-processing script: `preprocess_vqvae.py`


# Usage
Please see example commands in run.sh for training the vavae model.

Or you can use python train.py -m [model type]. The -m option can be used to tell the the script to train a different model.

[model type] can be:
- 'sys2': train original VQVAE
- 'sys3': train a self-supervised VQVAE with dual encoders
- 'sys4': train a semi-supervised VQVAE with dual encoders
- 'sys4a': train system 4a but use angular softmax
- 'sys5': train a semi-supervised VQVAE with dual encoders, and gradient reversal
- 'sys5a': train system 5a but use angular softmax


Please modify sampling rate and other parameters in [config.py](https://github.com/rhoposit/icassp2021/blob/main/config.py) before training.


# Pre-trained models
We offer monolingual VCTK pre-trained models for system 2 (sys2) and system 5 (sys5)


# Multi-gpu parallel training
Please see example commands in run_slurm.sh for running on SLURM with multiple GPUs

# Acknowledgement
The code is based on [mkotha/WaveRNN](https://github.com/mkotha/WaveRNN)

And is also based on [nii-yamagishilab/Extended_VQVAE](https://github.com/nii-yamagishilab/Extended_VQVAE)



This work was partially supported by the EPSRC Centre for DoctoralTraining in Data Science, funded by the UK Engineering and Physical Sci-ences Research Council (grant EP/L016427/1) and University of Edinburgh;and by a JST CREST Grant (JPMJCR18A6, VoicePersonae project), Japan.The numerical calculations were carried out on the TSUBAME 3.0 super-computer at the Tokyo Institute of Technology.

# License

MIT License
- Copyright (c) 2019, fatchord (https://github.com/fatchord)
- Copyright (c) 2019, mkotha (https://github.com/mkotha)
- Copyright (c) 2020, Yamagishi Laboratory, National Institute of Informatics.
- Copyright (c) 2021, Jennifer Williams and Yamagishi Laboratory, National Institute of Informatics.



Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.