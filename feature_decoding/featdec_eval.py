'''Feature decoding evaluation.'''


import argparse
from itertools import product
import os

from bdpy.dataform import Features, DecodedFeatures
from bdpy.evals.metrics import profile_correlation, pattern_correlation, pairwise_identification
import hdf5storage
import numpy as np
import pandas as pd
import yaml


# Main #######################################################################

def featdec_eval(
        decoded_feature_dir,
        true_feature_dir,
        output_file='./accuracy.pkl.gz',
        subjects=None,
        rois=None,
        features=None,
        feature_index_file=None,
        feature_decoder_dir=None,
        single_trial=False
):
    '''Evaluation of feature decoding.

    Input:

    - deocded_feature_dir
    - true_feature_dir

    Output:

    - output_file

    Parameters:

    TBA
    '''

    # Display information
    print('Subjects: {}'.format(subjects))
    print('ROIs:     {}'.format(rois))
    print('')
    print('Decoded features: {}'.format(decoded_feature_dir))
    print('')
    print('True features (Test): {}'.format(true_feature_dir))
    print('')
    print('Layers: {}'.format(features))
    print('')
    if feature_index_file is not None:
        print('Feature index: {}'.format(feature_index_file))
        print('')

    # Loading data ###########################################################

    # True features
    if feature_index_file is not None:
        features_test = Features(
            true_feature_dir, feature_index=feature_index_file)
    else:
        features_test = Features(true_feature_dir)

    # Decoded features
    decoded_features = DecodedFeatures(decoded_feature_dir)

    # Evaluating decoding performances #######################################

    if os.path.exists(output_file):
        print('Loading {}'.format(output_file))
        perf_df = pd.read_pickle(output_file)
    else:
        print('Creating an empty dataframe')
        perf_df = pd.DataFrame(columns=[
            'layer', 'subject', 'roi',
            'profile correlation', 'pattern correlation', 'identification accuracy'
        ])

    sample_types = ["slide0", "slide1", "slide2"]

    for layer, subject, roi in product(features, subjects, rois):
        print('Layer: {} - Subject: {} - ROI: {}'.format(layer, subject, roi))

        # Check if already calculated
        if len(perf_df.query(
                'layer == "{}" and subject == "{}" and roi == "{}"'.format(
                    layer, subject, roi
                )
        )) > 0:
            print('Already done. Skipped.')
            continue

        # Get predicted features
        pred_y = decoded_features.get(layer=layer, subject=subject, roi=roi)
        pred_labels = decoded_features.selected_label
        # Get true features
        true_labels = pred_labels
        true_y = features_test.get(layer=layer, label=true_labels)
        
        # Get train mean for normalization
        norm_param_dir = os.path.join(
            feature_decoder_dir,
            layer, subject, roi,
            'model'
        )
        train_y_mean = hdf5storage.loadmat(
            os.path.join(norm_param_dir, 'y_mean.mat'))['y_mean']
        train_y_std = hdf5storage.loadmat(
            os.path.join(norm_param_dir, 'y_norm.mat'))['y_norm']

        # Evaluation
        r_prof = profile_correlation(pred_y, true_y)
        r_patt = pattern_correlation(
            pred_y, true_y, mean=train_y_mean, std=train_y_std)
        print('Mean profile correlation:     {}'.format(np.nanmean(r_prof)))
        print('Mean pattern correlation:     {}'.format(np.nanmean(r_patt)))
        # Identification
        ident_list = []
        for slide in sample_types:
            print('Sample type: {}'.format(slide))
            sample_selector = np.array(
                [True if slide in tl else False for tl in true_labels])
            a_pred_y = pred_y[sample_selector, :]
            a_pred_labels = np.array(pred_labels)[sample_selector]
            ident_list.append(pairwise_identification(
                a_pred_y, true_y, single_trial=True, 
                pred_labels=a_pred_labels, true_labels=true_labels))
        ident = np.nanmean(np.vstack(ident_list), axis=0)
        print('Mean identification accuracy: {}'.format(np.nanmean(ident)))

        # Store results
        perf_df = perf_df.append(
            {
                'layer':   layer,
                'subject': subject,
                'roi':     roi,
                'profile correlation': r_prof.flatten(),
                'pattern correlation': r_patt.flatten(),
                'identification accuracy': ident.flatten(),
            },
            ignore_index=True
        )

    print(perf_df)

    # Save the results
    perf_df.to_pickle(output_file, compression='gzip')
    print('Saved {}'.format(output_file))

    print('All done')

    return output_file


# Entry point ################################################################

if __name__ == '__main__':
    #import sys
    #sys.argv = ["", "config/vggsound_fmriprep_rep4_500voxel_vggishish_allunits_fastl2lir_alpha100.yaml"]

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

    if 'analysis name' in conf:
        analysis_name = conf['analysis name']
    else:
        analysis_name = ''

    decoded_feature_dir = os.path.join(
        conf['decoded feature dir'],
        analysis_name,
        'decoded_features',
        conf['network']
    )

    if 'feature index file' in conf:
        feature_index_file = os.path.join(
            conf['training feature dir'][0], conf['network'], conf['feature index file'])
    else:
        feature_index_file = None

    if 'test single trial' in conf:
        single_trial = conf['test single trial']
    else:
        single_trial = False

    featdec_eval(
        decoded_feature_dir,
        os.path.join(conf['test feature dir'][0], conf['network']),
        output_file=os.path.join(decoded_feature_dir, 'accuracy.pkl.gz'),
        subjects=list(conf['test fmri'].keys()),
        rois=list(conf['rois'].keys()),
        features=conf['layers'],
        feature_index_file=feature_index_file,
        feature_decoder_dir=os.path.join(
            conf['feature decoder dir'],
            analysis_name,
            conf['network']
        ),
        single_trial=single_trial
    )
