#!/usr/bin/env bash
SRC=en
TGT=de
DATA_ROOT=/media/george/Data/mustc
vocab=8000
vtype=unigram

# ST
python prep_mustc_data.py \
  --data-root ${DATA_ROOT} --vocab-type $vtype --vocab-size $vocab \
  --langs $TGT