#!/bin/bash

WSI_fn_list_txt="/Jun_anonymized_dir/OvaryCancer/StromaReaction/WSIs/wsi_list.txt"

config_file="./config.ini"

# 0 delete all intermedia result, except reaction_prediction
# 1 keep all intermedia result
SAVE_INTERMEDIA=0
for WSI_fn in $(cat $WSI_fn_list_txt);
do
  echo processing $WSI_fn
  # extract patches from WSI
  python ./test/extract_patches.py --input=$WSI_fn --config=$config_file

  # tumor stroma segmentation
  python ./test/segmenation.py --config=$config_file

  # stroma reaction intensity estimation
  python ./test/stroma_reaction_prediction.py --config=$config_file

  if ! $SAVE_INTERMEDIA; then
    python ./test/clean_intermedia.py --config=$config_file
  fi
done



