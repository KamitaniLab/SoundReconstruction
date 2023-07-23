'''Feature decoding evaluation.'''


import argparse
from itertools import product
import os

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import euclidean

from bdpy.dataform import Features
from bdpy.evals.metrics import profile_correlation, pattern_correlation, pairwise_identification


# Functions #######################################################################


def get_euclidean_distance_matrix(xa, xb):
    """
    Calculate euclidean distance matrix by remove nan columns in each pair.
    """
    assert xa.shape[0] == xb.shape[0]
    if xa.ndim != 2:
        xa = xa.reshape(xa.shape[0], -1)
    if xb.ndim != 2:
        xb = xb.reshape(xb.shape[0], -1)

    dist_mat = np.zeros((xa.shape[0], xb.shape[0]))
    for i in range(xa.shape[0]):
        for j in range(xb.shape[0]):
            nan_col = np.logical_or(np.isnan(xa[i]), np.isnan(xb[j]))
            if np.sum(nan_col) == xa.shape[1]:
                dist_mat[i, j] = np.nan
            else:
                xa_ = xa[i][~nan_col]
                xb_ = xb[j][~nan_col]
                dist_mat[i, j] = euclidean(xa_, xb_)
    return dist_mat


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
            'layer', 'subject', 'roi', 'eval_feat',
        ])

    sample_types = ["slide0", "slide1", "slide2"]

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
            # Get true features
            true_labels = recon_labels[:]
            true_y = features_test.get(layer=eval_feat, label=true_labels)
            # Match the order of samples
            if not np.array_equal(recon_labels, true_labels):
                y_index = [np.where(np.array(true_labels) == x)[0][0] for x in recon_labels]
                true_y_sorted = true_y[y_index]
            else:
                true_y_sorted = true_y

            if audio_feature_mode:
                # Convert features
                if eval_feat == 'hnr':
                    pass
                elif eval_feat == 'sc':
                    recon_y = np.nanmedian(recon_y, axis=1).reshape(recon_y.shape[0], -1)
                    true_y_sorted = np.nanmedian(true_y_sorted, axis=1).reshape(recon_y.shape[0], -1)
                elif eval_feat == 'f0':
                    recon_y = np.nanmean(recon_y, axis=1).reshape(recon_y.shape[0], -1)
                    true_y_sorted = np.nanmean(true_y_sorted, axis=1).reshape(recon_y.shape[0], -1)
                # Calculate distance matrix
                dist_mat = get_euclidean_distance_matrix(recon_y, true_y_sorted)
                dist = dist_mat.diagonal()
                print('Mean euclidean distance: {}'.format(np.nanmean(dist)))
                # Identification
                ident_list = []
                for slide in sample_types:
                    print('Sample type: {}'.format(slide))
                    sample_selector = np.array([True if slide in tl else False for tl in true_labels])
                    a_dist_mat = dist_mat[np.ix_(sample_selector, sample_selector)]
                    ident = []
                    for i in range(a_dist_mat.shape[0]):
                        if np.isnan(a_dist_mat[i, i]):
                            ident.append(np.nan)
                        else:
                            win = np.array(a_dist_mat[i, i] < a_dist_mat[i], dtype=np.float32)
                            win[i] = np.nan
                            win[np.isnan(a_dist_mat[i])] = np.nan
                            ident.append(np.nanmean(win))
                    ident = np.array(ident)
                    ident_list.append(ident)
                print('Mean identification accuracy: {}'.format(np.nanmean(ident)))
                perf_df = perf_df.append(
                    {
                        'layer':   layer,
                        'subject': sbj,
                        'roi':     roi,
                        'eval_feat': eval_feat,
                        'euclidean distance': dist.flatten(),
                        'identification accuracy': ident.flatten(),
                    },
                    ignore_index=True
                )
            else:
                # Evaluation
                r_prof = profile_correlation(recon_y, true_y_sorted)
                r_patt = pattern_correlation(recon_y, true_y_sorted)
                ident = pairwise_identification(recon_y, true_y_sorted)
                print('Mean profile correlation:     {}'.format(np.nanmean(r_prof)))
                print('Mean pattern correlation:     {}'.format(np.nanmean(r_patt)))
                print('Mean identification accuracy: {}'.format(np.nanmean(ident)))
                perf_df = perf_df.append(
                    {
                        'layer':   layer,
                        'subject': sbj,
                        'roi':     roi,
                        'eval_feat': eval_feat,
                        'profile correlation': r_prof.flatten(),
                        'pattern correlation': r_patt.flatten(),
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
