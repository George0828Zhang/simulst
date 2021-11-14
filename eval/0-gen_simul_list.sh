#!/usr/bin/env bash
SPLIT=dev #tst-COMMON
EVAL_DATA=./data
. ../exp/data_path.sh

echo "segmenting ${SPLIT} data and calc gcmvn"
python ../DATA/seg_covost_data.py \
  --data-root ${DATA_ROOT} -s ${SRC} -t ${TGT} \
  --split ${SPLIT} --gcmvn-max-num 1500 \
  --output ${EVAL_DATA}

# echo "extracting global cmvn from train data"
# python ../DATA/get_covost_cmvn.py \
#   --data-root ${DATA_ROOT} -s ${SRC} -t ${TGT} \
#   --split train --gcmvn-max-num 1500 \
#   --output ${EVAL_DATA}