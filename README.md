# Sound Reconstruction

Data and demo code for [Jon-Yun Park et al., Sound reconstruction from human brain activity via a generative model with brain-like auditory features](https://arxiv.org/abs/2306.11629).

## Dataset

ここにfigshareとかのやつ


## Code
### Setup

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

### Download datasets and models
Download fmri_data and DNN feature files from figshare; 
see data/README.md
Download models from figshare;
see data/README.md 

### Usage


train_batch.sh

test_batch.sh <- evaluationいれよう

recon_batch.sh

recon_eval_batch.sh <- feature extractionも入れよう

最後に可視化してね
makefigure.ipynbファイル
