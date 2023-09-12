# coding: utf-8

"""
Reconstruction for VGG sound
"""

import os
import time
from itertools import product
import sys
import argparse
import yaml

import numpy as np
import torch
import soundfile
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import hdf5storage


# Functions ##########################################


def ResampleTemporal(item):
    feat_len = item.shape[3]
    if feat_len < 2:
        raise ValueError("val error")
    # evenly spaced points (abcdefghkl -> aoooofoooo)
    idx = np.linspace(0, feat_len, 21, dtype=np.int, endpoint=False)
    # xoooo xoooo -> ooxoo ooxoo
    shift = feat_len // (21 + 1)
    idx = idx + shift

    item = item[:, :, :, idx]
    return item


def recon_sound(
    lib_path,
    features_dir, features_decoders_dir,
    output_dir, model_dir,
    subjects, layers, rois, seed,
    test_average_num,
):
    '''
    Sound reconstruction with Transformer and Vocoder
    '''
    # Load libraries
    sys.path.insert(0, lib_path)
    from feature_extraction.demo_utils import load_model
    from sample_visualization import (spec_to_audio_to_st, tensor_to_plt)
    from bdpy.dataform import DecodedFeatures

    # Display information
    print('Subjects: {}'.format(subjects))
    print('ROIs:     {}'.format(rois))
    print('Layers:   {}'.format(layers))
    print('')
    print('Decoded features: {}'.format(features_dir))
    print('Feature decoders: {}'.format(features_decoders_dir))
    print('Output directory: {}'.format(output_dir))
    print('')

    # Load decoded features
    dec_features = DecodedFeatures(features_dir)

    for sbj, roi, layer in product(subjects, rois, layers):
        # Prepare output directory
        recon_output_dir = os.path.join(output_dir, layer, sbj, roi)
        if os.path.exists(recon_output_dir) == False:
            os.makedirs(recon_output_dir)

        model_name = layers[layer]['model']

        # Load models
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config, sampler, melgan, melception = load_model(
            model_name, model_dir, lib_path, device)

        # For scaling
        y_mean = hdf5storage.loadmat(os.path.join(
            features_decoders_dir, layer, sbj, roi, 'model', 'y_mean.mat'))["y_mean"]
        y_norm_cv = np.sqrt(test_average_num)

        dec_features.get(layer=layer, subject=sbj, roi=roi)
        dec_labels = dec_features.selected_label

        # initialize random seed
        torch.manual_seed(seed)

        # Loop for recon
        for i, label in enumerate(dec_labels):
            print("Recon target: {}".format(label))

            if os.path.exists(os.path.join(recon_output_dir, label + '.png')):
                print("Skip")
                continue

            # Load decoded feature
            feat = dec_features.get(
                layer=layer, subject=sbj, roi=roi, label=label)

            # Scaling
            feat = (feat - y_mean) * y_norm_cv + y_mean

            # Fill nan and inf with train mean
            ft_mask = np.isnan(feat)
            feat[ft_mask] = y_mean[ft_mask]
            ft_mask = np.isinf(feat)
            feat[ft_mask] = y_mean[ft_mask]

            # Reshape features
            visual_features = {}
            patch_size_i = 5
            patch_size_j = 21
            if "fc" in layer:
                visual_features['feature'] = np.array(
                    feat).reshape(1, -1, order='F')
            else:
                # feat = feat[np.newaxis]
                feat = ResampleTemporal(feat)
                visual_features['feature'] = np.array(feat).transpose(
                    3, 0, 1, 2).reshape([patch_size_j, -1])

            # Prepare Input
            batch = default_collate([visual_features])
            batch['feature'] = batch['feature'].to(device)
            c = sampler.get_input(sampler.cond_stage_key, batch)

            # Define Sampling Parameters
            W_scale = 1
            mode = 'full'
            temperature = 1.0
            top_x = sampler.first_stage_model.quantize.n_e // 2
            # use > 0 value, e.g. 15, to see the progress of generation (slows down the sampling speed)
            update_every = 0

            # Start sampling
            with torch.no_grad():
                start_t = time.time()

                quant_c, c_indices = sampler.encode_to_c(c)
                B, D, hr_h, hr_w = sampling_shape = (
                    1, 256, 5, patch_size_j * W_scale)

                if mode == 'full':
                    start_step = 0
                else:
                    start_step = (patch_size_j // 2) * patch_size_i

                z_pred_indices = torch.zeros((B, hr_h*hr_w)).long().to(device)

                pbar = tqdm(range(start_step, hr_w * hr_h),
                            desc='Sampling Codebook Indices')
                for step in pbar:
                    i = step % hr_h
                    j = step // hr_h
                    i_start = min(max(0, i - (patch_size_i // 2)),
                                  hr_h - patch_size_i)
                    j_start = min(max(0, j - (patch_size_j // 2)),
                                  hr_w - patch_size_j)
                    i_end = i_start + patch_size_i
                    j_end = j_start + patch_size_j
                    local_i = i - i_start
                    local_j = j - j_start

                    pbar.set_postfix(
                        Step=f'({i},{j}) | Local: ({local_i},{local_j}) | Crop: ({i_start}:{i_end},{j_start}:{j_end})'
                    )
                    patch = z_pred_indices.reshape(B, hr_w, hr_h).permute(0, 2, 1)[
                        :, i_start:i_end, j_start:j_end].permute(0, 2, 1).reshape(B, patch_size_i * patch_size_j)

                    # assuming we don't crop the conditioning and just use the whole c, if not desired uncomment the above
                    cpatch = c_indices
                    logits, _, attention = sampler.transformer(
                        patch[:, :-1], cpatch)

                    # remove conditioning
                    logits = logits[:, -patch_size_j * patch_size_i:, :]
                    local_pos_in_flat = local_j * patch_size_i + local_i
                    logits = logits[:, local_pos_in_flat, :]
                    logits = logits / temperature
                    logits = sampler.top_k_logits(logits, top_x)

                    # apply softmax to convert to probabilities
                    probs = torch.nn.functional.softmax(logits, dim=-1)

                    # sample from the distribution
                    ix = torch.multinomial(probs, num_samples=1)
                    z_pred_indices[:, j * hr_h + i] = ix

                    if update_every > 0 and step % update_every == 0:
                        z_pred_img = sampler.decode_to_img(
                            z_pred_indices, sampling_shape)
                        # fliping the spectrogram just for illustration purposes (low freqs to bottom, high - top)
                        z_pred_img_st = tensor_to_plt(
                            z_pred_img, flip_dims=(2,))

                # Show the final result
                z_pred_img = sampler.decode_to_img(
                    z_pred_indices, sampling_shape)
                z_pred_img_st = tensor_to_plt(z_pred_img, flip_dims=(2,))
                print(f'Sampling Time: {time.time() - start_t:3.2f} seconds')

                # Convert the reconstructed melspectrogram to wav data by Vocoder (MelGAN)
                waves = spec_to_audio_to_st(z_pred_img, config.data.params.spec_dir_path,
                                            config.data.params.sample_rate, show_griffin_lim=False,
                                            vocoder=melgan, show_in_st=False)
                print(
                    f'Sampling Time (with vocoder): {time.time() - start_t:3.2f} seconds')
                print(
                    f'Generated: {len(waves["vocoder"]) / config.data.params.sample_rate:.2f} seconds')

            # Save results
            save_mel_path = os.path.join(recon_output_dir, label + '.png')
            save_wav_path = os.path.join(recon_output_dir, label + '.wav')
            save_npy_path = os.path.join(recon_output_dir, label + '.npy')
            z_pred_img_st.savefig(save_mel_path)
            soundfile.write(
                save_wav_path, waves['vocoder'], config.data.params.sample_rate, 'PCM_24')
            np.save(save_npy_path, z_pred_img.cpu().detach().numpy())
            print(f'The sample has been saved @ {save_wav_path}')


if __name__ == '__main__':
    #import sys
    #sys.argv = ["", "config/recon_vggsound_attention_fmriprep_rep4_500voxel_vggishish_allunits_fastl2lir_alpha100.yaml"]
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

    feature_decoding_config = os.path.join(
        conf['feature decoding dir'], conf['feature decoding config'])
    with open(feature_decoding_config, 'r') as f:
        conf_featdec = yaml.safe_load(f)

    conf.update({
        'feature decoding': conf_featdec
    })

    if 'analysis name' in conf['feature decoding']:
        analysis_name = conf['feature decoding']['analysis name']
    else:
        analysis_name = ''

    features_dir = os.path.join(
        conf['feature decoding dir'],
        conf['feature decoding']['decoded feature dir'],
        analysis_name,
        'decoded_features',
        conf['feature decoding']['network']
    )

    features_decoders_dir = os.path.join(
        conf['feature decoding dir'],
        conf['feature decoding']['feature decoder dir'],
        analysis_name,
        conf['feature decoding']['network']
    )

    recon_sound(
        lib_path=conf['specvqgan dir'],
        features_dir=features_dir,
        features_decoders_dir=features_decoders_dir,
        output_dir=conf['recon output dir'],
        model_dir=conf['recon model dir'],
        subjects=conf['recon subjects'],
        layers=conf['recon layers'],
        rois=conf['recon rois'],
        seed=conf['seed'],
        test_average_num=conf['feature decoding']['test average num'],
    )
