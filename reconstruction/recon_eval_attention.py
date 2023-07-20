'''Feature decoding evaluation.'''


import argparse
from itertools import product
import os

import numpy as np
import pandas as pd
import yaml

from bdpy.dataform import Features
from bdpy.evals.metrics import profile_correlation, pattern_correlation, pairwise_identification


# Main #######################################################################


def recon_eval(
        recon_feature_dir,
        true_feature_dir,
        output_file,
        subjects,
        rois,
        layers,
):
    '''Evaluation of feature decoding.'''

    # Display information
    print('Subjects: {}'.format(subjects))
    print('ROIs:     {}'.format(rois))
    print('Layers:     {}'.format(layers))
    print('')
    print('Recon features: {}'.format(recon_feature_dir))
    print('True features (Test): {}'.format(true_feature_dir))
    print('')
    # Evaluation features
    eval_features = [
        'conv1', 'conv2', 'conv3', 'conv4', 'conv5',
        'f0', 'fc1', 'hnr', 'mel',
        'mix5_b', 'mix5_c', 'mix5_d',
        'mix6_a', 'mix6_b', 'mix6_c', 'mix6_d', 'mix6_e',
        'mix7_a', 'mix7_b', 'mix7_c', 'sc'
    ]
    print('Eval features: {}'.format(eval_features))
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
            'profile correlation for attend', 'profile correlation for unattend', 
            'pattern correlation for attend', 'pattern correlation for unattend', 
            'identification accuracy'
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
            # Get true features
            true_labels = recon_labels[:]
            true_y = features_test.get(layer=eval_feat, label=true_labels)
            # Match the order of samples
            if not np.array_equal(recon_labels, true_labels):
                y_index = [np.where(np.array(true_labels) == x)[0][0] for x in recon_labels]
                true_y_sorted = true_y[y_index]
            else:
                true_y_sorted = true_y

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

    recon_eval(
        recon_feature_dir=conf['eval feat output dir'],
        true_feature_dir=conf['eval true feature dir'],
        output_file=os.path.join(conf['eval feat output dir'], 'quality.pkl.gz'),
        subjects=conf['recon subjects'],
        rois=conf['recon rois'],
        layers=list(conf['recon layers'].keys()),
    )
