#!/usr/bin/env bash
. ./data_path.sh
SRC=pho  # replace phone src set instead
DECODED=./distilled/generate-train.txt
TRAIN=${DATA}/train_st_${SRC}_${TGT}.tsv
OUT=${DATA}/distill_st_${SRC}_${TGT}.tsv

python ../DATA/create_distillation_tsv.py \
    --train-file ${TRAIN} \
    --distill-file ${DECODED} \
    --out-file ${OUT} \
    --detok-lang ${TGT} \
    --verbose 