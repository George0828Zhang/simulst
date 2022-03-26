#!/usr/bin/env bash
export TGT=de
SPLIT=tst-COMMON
EVAL_DATA=./data_${TGT}
. ../exp/data_path.sh
MUSTC_ROOT=`dirname ${DATA}`
DATA=$(realpath ../DATA)
export PYTHONPATH="$DATA:$DATA/mustc:$PYTHONPATH"
echo "segmenting ${SPLIT} data"
python -m mustc.seg_mustc_data \
  --data-root ${MUSTC_ROOT} --lang ${TGT} \
  --split ${SPLIT} \
  --thresholds 20,40,60 \
  --output ${EVAL_DATA}
