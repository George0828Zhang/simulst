#!/usr/bin/env bash
SPLIT=dev #tst-COMMON
EVAL_DATA=./data
. ../exp/data_path.sh

echo "segmenting ${SPLIT} data"
python ../DATA/seg_covost_data.py \
  --data-root ${DATA_ROOT} -s ${SRC} -t ${TGT} \
  --split ${SPLIT} \
  --output ${EVAL_DATA}

echo "tokenize..."
python ../DATA/text_processors.py zh zh \
  <${EVAL_DATA}/${SPLIT}.${TGT} >${EVAL_DATA}/${SPLIT}.${TGT}.tok

echo "copying global cmvn from train data"
cp ${DATA}/gcmvn.npz ${EVAL_DATA}
