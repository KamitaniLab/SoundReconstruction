# Sound Reconstruction

Data and demo code for [Jon-Yun Park et al., Sound reconstruction from human brain activity via a generative model with brain-like auditory features](https://arxiv.org/abs/2306.11629).

## Preparation
### Prepare environment
Create conda environment.

```
conda env create --name specvqgan -f specvqgan.yaml 
python -c "import torch; print(torch.cuda.is_available())"
# True
```

Clone SpecVQGAN repository under "SoundReconstruction" directory.
Transformerのパスを書き換えているため，Originalのリポジトリではなく，こちらのforkしたリポジトリを使用してください．

```
# cd to "SoundReconstruction"
git clone git@github.com:KamitaniLab/SpecVQGAN.git
```

### Download datasets
Download fmri_data and DNN feature files from figshare; 
see data/README.md

### Download models
Download models from figshare;
see data/README.md 

## Usage

