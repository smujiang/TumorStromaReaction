#!/usr/bin/bash

config_file="./test_config_seg.ini"

# 0 delete all intermedia result, except reaction_prediction
# 1 keep all intermedia result
SAVE_INTERMEDIA=False

# extract patches from WSI
#python ./test/extract_patches.py --config=$config_file

# tumor stroma segmentation
python ./test/segmenation.py --config=$config_file

# stroma reaction intensity estimation
#python ./test/reaction_prediction.py --config=$config_file

#if ! $SAVE_INTERMEDIA; then
#  python ./test/clean_intermedia.py --config=$config_file
#fi
