#!/usr/bin/env bash
SPLIT=dev #tst-COMMON
EVAL_DATA=./data
. ../exp/data_path.sh
MUSTC_ROOT=`dirname ${DATA}`

echo "segmenting ${SPLIT} data"
python ../DATA/seg_mustc_data.py \
  --data-root ${MUSTC_ROOT} --lang ${TGT} \
  --split ${SPLIT} \
  --output ${EVAL_DATA}

echo "extracting global cmvn from train data"
python ../DATA/get_mustc_cmvn.py \
  --data-root ${MUSTC_ROOT} --lang ${TGT} \
  --split train --gcmvn-max-num 1500 \
  --output ${EVAL_DATA}