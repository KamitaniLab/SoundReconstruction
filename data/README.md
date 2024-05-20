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
- `models`: DNN models

### Data files

```
data
├── features
│   ├── VGGishishTest
│   │   ├── melception
│   │   └── vggishish
│   └── VGGishishTrain
│       └── vggishish
├── fmri_data
│   ├── S1_VGGishishAttentionTest_volume_native.h5
│   ├── S1_VGGishishTest_volume_native.h5
│   ├── S1_VGGishishTrain_volume_native.h5
│   ├── S2_VGGishishAttentionTest_volume_native.h5
│   ├── S2_VGGishishTest_volume_native.h5
│   ├── S2_VGGishishTrain_volume_native.h5
│   ├── S3_VGGishishAttentionTest_volume_native.h5
│   ├── S3_VGGishishTest_volume_native.h5
│   ├── S3_VGGishishTrain_volume_native.h5
│   ├── S4_VGGishishAttentionTest_volume_native.h5
│   ├── S4_VGGishishTest_volume_native.h5
│   ├── S4_VGGishishTrain_volume_native.h5
│   ├── S5_VGGishishAttentionTest_volume_native.h5
│   ├── S5_VGGishishTest_volume_native.h5
│   └── S5_VGGishishTrain_volume_native.h5
└── models
    └── melception
    │   └── melception-21-05-10T09-28-40.pt
    └── specvqgan
        ├── 2022-10-28T10-52-39_transformer_5_3_final
        ├── 2023-09-01T10-35-12_transformer_fc3_final
        ├── 2023-11-16T15-41-28_transformer_4_1_final
        ├── 2023-11-16T15-41-57_transformer_3_1_final
        ├── 2023-11-16T15-42-33_transformer_2_1_final
        └── 2023-11-16T15-43-27_transformer_1_1_final
```

## Stimulus sound clip

The sound clips used in the experiment were selected from the [VGG sound dataset](https://www.robots.ox.ac.uk/~vgg/data/vggsound/). The Youtube video ID, retrieval time, and label information are listed in [SoundReconstruction_stimulus.csv](SoundReconstruction_stimulus.csv).
