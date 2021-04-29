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
python prep_mustc_data.py \
  --data-root ${DATA_ROOT} --vocab-type $vtype --vocab-size $vocab \
  --langs $TGT