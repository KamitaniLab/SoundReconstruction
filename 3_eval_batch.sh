#!/bin/sh
cd feature_decoding
python featdec_eval.py config/vggsound_fmriprep_rep4_500voxel_vggishish_allunits_fastl2lir_alpha100.yaml
python featdec_eval_attention.py config/vggsound_attention_fmriprep_rep4_500voxel_vggishish_allunits_fastl2lir_alpha100.yaml