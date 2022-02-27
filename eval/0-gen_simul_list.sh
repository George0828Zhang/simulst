#!/usr/bin/env bash
SPLIT=dev #tst-COMMON
EVAL_DATA=./data
. ../exp/data_path.sh
MUSTC_ROOT=`dirname ${DATA}`
export PYTHONPATH="$(realpath ../DATA):$PYTHONPATH"
echo "segmenting ${SPLIT} data"
python -m mustc.seg_mustc_data \
  --data-root ${MUSTC_ROOT} --lang ${TGT} \
  --split ${SPLIT} \
  --output ${EVAL_DATA}
