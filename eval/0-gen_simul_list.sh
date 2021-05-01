#!/usr/bin/env bash
SPLIT=dev #tst-COMMON
EVAL_DATA=./data
. ../exp/data_path.sh
MUSTC_ROOT=`dirname ${DATA}`

python ../DATA/seg_mustc_data.py \
  --data-root ${MUSTC_ROOT} --lang ${TGT} \
  --split ${SPLIT} --task st \
  --output ${EVAL_DATA}