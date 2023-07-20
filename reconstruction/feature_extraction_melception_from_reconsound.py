#coding:utf-8


import sys
import os
import time
import random
from itertools import product
from tqdm import tqdm
from pathlib import Path
import argparse
import yaml 
import glob

import soundfile
import numpy as np
from scipy.io import loadmat, savemat
import scipy.signal
import torch
from torch.utils.data.dataloader import DataLoader, default_collate
from torchvision.models.inception import BasicConv2d, Inception3
import torchvision
import librosa
import Signal_Analysis.features.signal as sig
from matplotlib import pyplot as plt
import PIL
from omegaconf import OmegaConf

from bdpy.dl.torch import FeatureExtractor


## Functions
# TODO: refactor  ./evaluation/feature_extractors/melception.py to handle this class as well.
# So far couldn't do it because of the difference in outputs
class Melception(Inception3):
    def __init__(self, num_classes, **kwargs):
        # inception = Melception(num_classes=309)
        super().__init__(num_classes=num_classes, **kwargs)
        # the same as https://github.com/pytorch/vision/blob/5339e63148/torchvision/models/inception.py#L95
        # but for 1-channel input instead of RGB.
        self.Conv2d_1a_3x3 = BasicConv2d(1, 32, kernel_size=3, stride=2)
        # also the 'hight' of the mel spec is 80 (vs 299 in RGB) we remove all max pool from Inception
        self.maxpool1 = torch.nn.Identity()
        self.maxpool2 = torch.nn.Identity()

    def forward(self, x):
        x = x.unsqueeze(1)
        return super().forward(x)
    
def extract_feat_from_recon_sound(
    lib_path,
    base_input_dir, 
    base_output_dir,
    model_dir,
    eval_model_file,
    subjects,
    layers,
    rois, 
    ):
    
    # Load library
    sys.path.insert(0, lib_path)
    from feature_extraction.demo_utils import (extract_melspectrogram, load_model)
    from sample_visualization import (get_class_preditions, tensor_to_plt)
    from specvqgan.data.vggsound import CropImage
    from specvqgan.util import get_ckpt_path

    def ScalingLayer(inp, lib_path, device):
        # we are gonna use get_ckpt_path to donwload the stats as well
        stat_path = get_ckpt_path('vggishish_mean_std_melspec_10s_22050hz', os.path.join(lib_path, 'specvqgan/modules/autoencoder/lpaps'))
        # if for images we normalize on the channel dim, in spectrogram we will norm on frequency dimension
        means, stds = np.loadtxt(stat_path, dtype=np.float32).T
        # the normalization in means and stds are given for [0, 1], but specvqgan expects [-1, 1]:
        means = 2 * means -1
        stds = 2 * stds 
        # input is expected to be (B, 1, F, T)
        shift = torch.from_numpy(means)[None, None, :, None].to(device)
        scale = torch.from_numpy(stds)[None, None, :, None].to(device)
        return (inp - shift) / scale
    
    # Set melception
    num_classes_vggsound = 309
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = Melception(num_classes=num_classes_vggsound)
    encoder.to(device)
    ckpt = torch.load(eval_model_file, map_location='cpu')
    encoder.load_state_dict(ckpt['model'],strict=False)
    print((f'The model was trained for {ckpt["epoch"]} epochs. Loss: {ckpt["loss"]:.4f}'))
    encoder.eval()
    eval_layer_mapping = {
        'conv1': 'Conv2d_1a_3x3',
        'conv2': 'Conv2d_2a_3x3',
        'conv3': 'Conv2d_2b_3x3',
        'conv4': 'Conv2d_3b_1x1',
        'conv5': 'Conv2d_4a_3x3',
        'mix5_b': 'Mixed_5b',
        'mix5_c': 'Mixed_5c',
        'mix5_d': 'Mixed_5d',
        'mix6_a': 'Mixed_6a',
        'mix6_b': 'Mixed_6b',
        'mix6_c': 'Mixed_6c',
        'mix6_d': 'Mixed_6d',
        'mix6_e': 'Mixed_6e',
        'mix7_a': 'Mixed_7a',
        'mix7_b': 'Mixed_7b',
        'mix7_c': 'Mixed_7c',
        'fc1': 'fc',
    }
    eval_layers = list(eval_layer_mapping.keys())
    
    # Set feature extractor
    feature_extractor = FeatureExtractor(encoder, eval_layers, eval_layer_mapping, device=device, detach=True)

    # Load models
    model_name = layers[next(iter(layers))]['model'] # とりあえず適当にとってくる
    config, sampler, melgan, melception = load_model(model_name, model_dir, lib_path, device)

    featlist = ['f0','sc','hnr']
    for sbj,roi, layer in product(subjects, rois, layers):
        input_dir = os.path.join(base_input_dir, layer, sbj, roi)
        output_dir = os.path.join(base_output_dir, layer, sbj, roi)
        if os.path.exists(output_dir)==False:
            os.makedirs(output_dir)
            
        filelist = sorted(glob.glob(os.path.join(input_dir, "*.wav")))
        recon_labels = [os.path.splitext(os.path.split(f)[1])[0] for f in filelist]

        for i, recon_label in enumerate(recon_labels):
            print(recon_label)   
            if os.path.exists(os.path.join(output_dir, "hnr", recon_label + ".mat")):
                print("Skip")
                continue
            else:
                # Prepare melspectrogram
                video_path = os.path.join(input_dir, recon_label + ".wav")
                audio_fps = 22050
                spectrogram = extract_melspectrogram(video_path, audio_fps, duration=4)
                spectrogram = {'input': spectrogram}
                random_crop = False
                crop_img_fn = CropImage([config.data.params.mel_num, 336], random_crop)
                spectrogram = crop_img_fn(spectrogram)

                batch = default_collate([spectrogram])
                batch['image'] = batch['input'].to(device)
                x = sampler.get_input(sampler.first_stage_key, batch)

                # Save melspectrogram
                save_dir = os.path.join(output_dir, "mel")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                savemat(os.path.join(save_dir, recon_label + '.mat'), {"feat": x.cpu().detach().numpy()})
                
                # Extract melception features
                x = ScalingLayer(x, lib_path, device)
                features = feature_extractor.run(x.squeeze(0))
                
                # Save melception features
                for k, v in features.items():
                    save_dir = os.path.join(output_dir, k)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    savemat(os.path.join(save_dir, recon_label + '.mat'), {"feat": v})

                # Calculate audio features
                v = {}
                y, sr = librosa.load(video_path)
                if len(y) < 4 * sr:
                    y = np.concatenate((y, np.zeros(4 * sr - len(y))))
                f0, vf, vp = librosa.pyin(y[:sr * 4], fmin=125, fmax=7600)
                v['f0'] = f0
                v['sc'] = librosa.feature.spectral_centroid(y[:sr * 4])
                v['hnr'] = HNR_true=sig.get_HNR(y[:sr * 4], sr)

                # Save audio features
                for k in featlist:
                    save_dir = os.path.join(output_dir, k)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    savemat(os.path.join(save_dir, recon_label + '.mat'), {"feat": v[k]})

if __name__ == "__main__":
  #import sys
  #sys.argv = ["", "config/recon_vggsound_fmriprep_rep4_500voxel_vggishish_allunits_fastl2lir_alpha100.yaml"]
  
  parser = argparse.ArgumentParser()
  parser.add_argument(
    'conf',
    type=str,
    help='analysis configuration file',
  )
  args = parser.parse_args()

  conf_file = args.conf

  with open(conf_file, 'r') as f:
    conf = yaml.safe_load(f)

  conf.update({
    '__filename__': os.path.splitext(os.path.basename(conf_file))[0]
  })
  
  extract_feat_from_recon_sound(
    lib_path = conf['specvqgan dir'],
    base_input_dir = conf['recon output dir'], 
    base_output_dir = conf['eval feat output dir'],
    model_dir = conf['recon model dir'],
    eval_model_file = conf['eval model file'],
    subjects=conf['recon subjects'],
    layers=conf['recon layers'],
    rois=conf['recon rois'],  
    )
  
  
