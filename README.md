# Sound Reconstruction

This repository shows the data and demo code for [Jong-Yun Park et al., Sound reconstruction from human brain activity via a generative model with brain-like auditory features](https://arxiv.org/abs/2306.11629).

## Dataset

- Raw fMRI data: TBA
- Preprocessed fMRI data, DNN features extracted from sound clips: [figshare](https://figshare.com/articles/dataset/23633751)
- Trained transformer models: [figshare](https://figshare.com/articles/dataset/23633751)
- Stimulus sound clips: Refer to [data/README.md](data/README.md) .

## Code

### Setup

1. Clone this `SoundReconstruction` repository to your local machine (GPU machine preferred).
```
git clone git@github.com:KamitaniLab/SoundReconstruction.git
```

2. Create conda environment using the `specvqgan.yaml`.
```
conda env create --name specvqgan -f specvqgan.yaml 
python -c "import torch; print(torch.cuda.is_available())"
# True
```

3. Clone `SpecVQGAN` repository next to `SoundReconstruction` directory. Please use the following fork repository instead of [the original SpecVQGAN repository](https://github.com/v-iashin/SpecVQGAN) because the path of the Transformer configuration file has been rewritten.
```
git clone git@github.com:KamitaniLab/SpecVQGAN.git
```

### Download datasets and models

See [data/README.md](data/README.md).

### Usage

We provide scripts that reproduce main results in the original paper.
Please execute the sh files in the following order.

1. Train feature decoders to predict the VGGishish features. 
```
./1_train_batch.sh
```

2. Using the decoders trained in step.1, perform feature predictions. (Perform the prediction for the attention task dataset at the same time.)
```
./2_test_batch.sh
```

3. Validate the prediction accuracy of predicted features.
```
./3_eval_batch.sh
```
Visualize the prediction accuracy with the following notebook. This notebook draws Fig.3D and Fig.3E of the original paper.
```
feature_decoding/makefigures_featdec_eval.ipynb
```

4. Reconstruct sound clips using predicted features.
```
./4_recon_batch.sh
```

5. Validate the quality of reconstructed sound.
```
./5_recon_eval_batch.sh 
```
Visualize the reconstruction quality with the following notebooks. These notebooks draws Fig.4C and Fig.8C of the original paper.
```
reconstruction/makefigures_recon_eval.ipynb
reconstruction/makefigures_recon_eval_attention.ipynb
```
