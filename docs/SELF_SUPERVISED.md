# Training the self-supervised models

## Installation

Clone [the repository](https://github.com/bhigy/ZeroSpeech) containing the code for the self-supervised model (based on
[bshall/ZeroSpeech](https://github.com/bshall/ZeroSpeech)) and cd into it:

```sh
git clone https://github.com/bhigy/ZeroSpeech.git bshall-zerospeech
cd bshall-zerospeech
```

Install [NVIDIA/apex](https://github.com/NVIDIA/apex#quick-start) for mixed precision training.

Add missing dependencies to the conda environment:

```sh
pip install librosa==0.7.2 numba==0.48 tqdm==4.45.0 hydra-core==0.11.3 pyloudnorm==0.1.0 tensorboard==2.2.1
```

## Dataset

Instructions explaining how to download and preprocess the Zerospeech 2020 dataset can be found in the self-supervised model's [original
documentation](https://github.com/bhigy/ZeroSpeech#data-and-preprocessing).


## Training

To train the self-supervised models, simply run:

```bash
./train_size.sh
```
