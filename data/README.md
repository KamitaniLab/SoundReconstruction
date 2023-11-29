# Sound Reconstruction Dataset

## Downloading dataset

First, please install [bdpy](https://github.com/KamitaniLab/bdpy) via pip (version >= 0.21).

```
$ pip install bdpy
```

Then you can download data with the following command.

```
$ python download.py <target>
```

Targets:

- `fmri_train`: fMRI data for training
- `fmri_test`: fMRI data for test
- `fmri_attention`: fMRI data for attention experiment
- `features_train`: DNN features for training
- `features_test`: DNN features for test
- `model`: DNN models
