'''Feature decoding evaluation.'''


import argparse
from itertools import product
import os

import numpy as np
import pandas as pd
import yaml

from bdpy.dataform import Features
from bdpy.evals.metrics import profile_correlation, pattern_correlation


# Main #######################################################################


def recon_eval(
        recon_feature_dir,
        true_feature_dir,
        output_file,
        eval_features,
        audio_feature_mode,
        subjects,
        rois,
        layers,
):
    '''Evaluation of feature decoding.'''

    # Display information
    print('Audio feat mode:{}'.format(audio_feature_mode))
    print('Subjects: {}'.format(subjects))
    print('ROIs:     {}'.format(rois))
    print('Layers:   {}'.format(layers))
    print('')
    print('Recon features:       {}'.format(recon_feature_dir))
    print('True features (Test): {}'.format(true_feature_dir))
    print('Evaluation features:  {}'.format(eval_features))
    print('')

    # Loading data ###########################################################
    # Get true features
    features_test = Features(true_feature_dir)

    # Evaluating decoding performances #######################################
    if os.path.exists(output_file):
        print('Loading {}'.format(output_file))
        perf_df = pd.read_pickle(output_file)
    else:
        print('Creating an empty dataframe')
        perf_df = pd.DataFrame(columns=[
            'layer', 'subject', 'roi', 'eval_feat'
        ])

    for sbj, layer, roi in product(subjects, layers, rois):
        print('Subject: {} - Layer: {} - ROI: {}'.format(sbj, layer, roi))
        for eval_feat in eval_features:
            print('Eval feat: {}'.format(eval_feat))

            # Check if already calculated
            if len(perf_df.query(
                    'layer == "{}" and subject == "{}" and roi == "{}" and eval_feat == "{}"'.format(
                        layer, sbj, roi, eval_feat
                    )
            )) > 0:
                print('Already done. Skipped.')
                continue

            # Get recon features for evaluation
            recon_features = Features(os.path.join(recon_feature_dir, layer, sbj, roi))
            recon_y = recon_features.get(layer=eval_feat)
            recon_labels = recon_features.labels
            attend_recon_labels = [pl.split("_unattend_")[0].split("attend_")[1] for pl in recon_labels]
            unattend_recon_labels = [pl.split("_unattend_")[1] for pl in recon_labels]
            # Get true features
            true_labels = attend_recon_labels + unattend_recon_labels
            true_labels = np.unique(true_labels)
            true_y = features_test.get(layer=eval_feat, label=true_labels)
            attend_index = [np.where(np.array(true_labels) == x)[0][0] for x in attend_recon_labels]
            attend_true_features = true_y[attend_index]
            unattend_index = [np.where(np.array(true_labels) == x)[0][0] for x in unattend_recon_labels]
            unattend_true_features = true_y[unattend_index]

            if audio_feature_mode:
                # Convert features
                if eval_feat == 'hnr':
                    pass
                elif eval_feat == 'sc':
                    recon_y = np.nanmedian(recon_y, axis=1)
                    attend_true_features = np.nanmedian(attend_true_features, axis=1)
                    unattend_true_features = np.nanmean(unattend_true_features, axis=1)
                elif eval_feat == 'f0':
                    recon_y = np.nanmean(recon_y, axis=1)
                    attend_true_features = np.nanmean(attend_true_features, axis=1)
                    unattend_true_features = np.nanmean(unattend_true_features, axis=1)
                # Evaluation
                attend_d = np.abs(recon_y - attend_true_features)
                unattend_d = np.abs(recon_y - unattend_true_features)
                # Attend v.s. Unattend identification
                nan_rows = np.logical_or(np.isnan(attend_d), np.isnan(unattend_d))
                ident = (attend_d < unattend_d).astype(np.float32)
                ident[nan_rows] = np.nan
                print('Mean euclidean distance for attend stimuli:   {}'.format(np.nanmean(attend_d)))
                print('Mean euclidean distance for unattend stimuli: {}'.format(np.nanmean(unattend_d)))
                print('Mean identification accuracy: {}'.format(np.nanmean(ident)))
                perf_df = perf_df.append(
                    {
                        'layer':   layer,
                        'subject': sbj,
                        'roi':     roi,
                        'eval_feat': eval_feat,
                        'distance for attend': attend_d.flatten(),
                        'distance for unattend': unattend_d.flatten(),
                        'identification accuracy': ident.flatten(),
                    },
                    ignore_index=True
                )
            else:
                # Evaluation
                attend_r_prof = profile_correlation(recon_y, attend_true_features)
                unattend_r_prof = profile_correlation(recon_y, unattend_true_features)
                attend_r_patt = pattern_correlation(recon_y, attend_true_features)
                unattend_r_patt = pattern_correlation(recon_y, unattend_true_features)
                # Attend v.s. Unattend identification
                ident = attend_r_patt > unattend_r_patt
                ident = ident.astype(np.float32)
                print('Mean profile correlation for attend stimuli:   {}'.format(np.nanmean(attend_r_prof)))
                print('Mean profile correlation for unattend stimuli: {}'.format(np.nanmean(unattend_r_prof)))
                print('Mean pattern correlation for attend stimuli:   {}'.format(np.nanmean(attend_r_patt)))
                print('Mean pattern correlation for unattend stimuli: {}'.format(np.nanmean(unattend_r_patt)))
                print('Mean identification accuracy: {}'.format(np.nanmean(ident)))
                perf_df = perf_df.append(
                    {
                        'layer':   layer,
                        'subject': sbj,
                        'roi':     roi,
                        'eval_feat': eval_feat,
                        'profile correlation for attend': attend_r_prof.flatten(),
                        'profile correlation for unattend': unattend_r_prof.flatten(),
                        'pattern correlation for attend': attend_r_patt.flatten(),
                        'pattern correlation for unattend': unattend_r_patt.flatten(),
                        'identification accuracy': ident.flatten(),
                    },
                    ignore_index=True
                )

    # Display
    print(perf_df)

    # Save the results
    perf_df.to_pickle(output_file, compression='gzip')
    print('Saved {}'.format(output_file))

    print('All done')
    return output_file


# Entry point ################################################################

if __name__ == '__main__':
    #import sys
    #sys.argv = ["", "config/recon_vggsound_attention_fmriprep_rep4_500voxel_vggishish_allunits_fastl2lir_alpha100.yaml"]

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

    # Evaluation for audiofeatures
    eval_features = [
        'f0', 'hnr', 'sc'
    ]
    recon_eval(
        recon_feature_dir=conf['eval feat output dir'],
        true_feature_dir=conf['eval true feature dir'],
        output_file=os.path.join(conf['eval feat output dir'], 'quality_audiofeature.pkl.gz'),
        eval_features=eval_features,
        audio_feature_mode=True,
        subjects=conf['recon subjects'],
        rois=conf['recon rois'],
        layers=list(conf['recon layers'].keys()),
    )

    # Evaluation for Melception layer features
    eval_features = [
        'conv1', 'conv2', 'conv3', 'conv4', 'conv5',
        'fc1', 'mel',
        'mix5_b', 'mix5_c', 'mix5_d',
        'mix6_a', 'mix6_b', 'mix6_c', 'mix6_d', 'mix6_e',
        'mix7_a', 'mix7_b', 'mix7_c'
    ]
    recon_eval(
        recon_feature_dir=conf['eval feat output dir'],
        true_feature_dir=conf['eval true feature dir'],
        output_file=os.path.join(conf['eval feat output dir'], 'quality.pkl.gz'),
        eval_features=eval_features,
        audio_feature_mode=False,
        subjects=conf['recon subjects'],
        rois=conf['recon rois'],
        layers=list(conf['recon layers'].keys()),
    )

