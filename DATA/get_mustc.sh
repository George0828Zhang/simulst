#!/usr/bin/env bash
SRC=en
TGT=de
DATA_ROOT=/media/george/Data/mustc
vocab=8000
vtype=unigram

FAIRSEQ=../
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
source ~/envs/apex/bin/activate

# ST
feats=${DATA_ROOT}/${SRC}-${TGT}/fbank80.zip
if [ -f ${feats} ]; then
  echo "${feats} already exists. It is likely that you set the wrong language which is already processed."
  echo "Please change data root or clear ${feats} before continuing."
  echo "Alternatively uncomment the command below to re-process manifest only."
  # python prep_mustc_data.py \
  #   --data-root ${DATA_ROOT} --vocab-type $vtype --vocab-size $vocab \
  #   --langs $TGT --manifest-only
else
  echo "processing ${DATA_ROOT}/${SRC}-${TGT}"
  python prep_mustc_data.py \
    --data-root ${DATA_ROOT} --vocab-type $vtype --vocab-size $vocab \
    --langs $TGT
fi