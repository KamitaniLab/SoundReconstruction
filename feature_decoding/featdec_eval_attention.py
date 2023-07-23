'''Feature decoding evaluation for attention task.'''


import argparse
from itertools import product
import os

from bdpy.dataform import Features, DecodedFeatures
from bdpy.evals.metrics import profile_correlation, pattern_correlation
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
        features_test = Features(true_feature_dir, feature_index=feature_index_file)
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
            'profile correlation for attend', 'profile correlation for unattend',
            'pattern correlation for attend', 'pattern correlation for unattend',
            'identification accuracy',
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
        pred_y = pred_y.reshape(pred_y.shape[0], -1)
        pred_labels = decoded_features.selected_label
        attend_pred_labels = [pl.split("_unattend_")[0].split("attend_")[1] for pl in pred_labels]
        unattend_pred_labels = [pl.split("_unattend_")[1] for pl in pred_labels]
        # Get true features
        true_labels = attend_pred_labels + unattend_pred_labels
        true_labels = np.unique(true_labels)
        true_y = features_test.get(layer=layer, label=true_labels)
        true_y = true_y.reshape(true_y.shape[0], -1)
        attend_index = [np.where(np.array(true_labels) == x)[0][0] for x in attend_pred_labels]
        attend_true_features = true_y[attend_index]
        unattend_index = [np.where(np.array(true_labels) == x)[0][0] for x in unattend_pred_labels]
        unattend_true_features = true_y[unattend_index]
        # Load Y mean and SD
        norm_param_dir = os.path.join(
            feature_decoder_dir,
            layer, subject, roi,
            'model'
        )
        train_y_mean = hdf5storage.loadmat(os.path.join(norm_param_dir, 'y_mean.mat'))['y_mean']
        train_y_std = hdf5storage.loadmat(os.path.join(norm_param_dir, 'y_norm.mat'))['y_norm']

        # Evaluation
        attend_r_prof = profile_correlation(pred_y, attend_true_features)
        unattend_r_prof = profile_correlation(pred_y, unattend_true_features)
        attend_r_patt = pattern_correlation(pred_y, attend_true_features, mean=train_y_mean, std=train_y_std)
        unattend_r_patt = pattern_correlation(pred_y, unattend_true_features, mean=train_y_mean, std=train_y_std)
        print('Mean profile correlation for attend stimuli:   {}'.format(np.nanmean(attend_r_prof)))
        print('Mean profile correlation for unattend stimuli: {}'.format(np.nanmean(unattend_r_prof)))
        print('Mean pattern correlation for attend stimuli:   {}'.format(np.nanmean(attend_r_patt)))
        print('Mean pattern correlation for unattend stimuli: {}'.format(np.nanmean(unattend_r_patt)))
        # Attend v.s. Unattend identification
        ident_list = []
        for slide in sample_types:
            print('Sample type: {}'.format(slide))
            sample_selector = np.array([True if slide in al else False for al in attend_pred_labels])
            a_attend_r_patt = attend_r_patt[sample_selector]
            a_unattend_r_patt = unattend_r_patt[sample_selector]
            a_ident = (a_attend_r_patt > a_unattend_r_patt).astype(np.float32)
            ident_list.append(a_ident)
        ident = np.nanmean(np.vstack(ident_list), axis=0)
        nan_rows = np.isnan(ident)
        ident = (ident > 0.5).astype(np.float32)
        ident[nan_rows] = np.nan
        print('Mean identification accuracy: {}'.format(np.nanmean(ident)))
        perf_df = perf_df.append(
            {
                'layer':   layer,
                'subject': subject,
                'roi':     roi,
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
    import sys
    sys.argv = ["", "config/vggsound_attention_fmriprep_rep4_500voxel_vggishish_allunits_fastl2lir_alpha100.yaml"]

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
        feature_index_file = os.path.join(conf['training feature dir'][0], conf['network'], conf['feature index file'])
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
