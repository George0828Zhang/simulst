#!/usr/bin/env bash
export TGT=de
. ./data_path.sh
DECODED=./distilled_${TGT}/generate-test.txt
TRAIN=${DATA}/train_st.tsv
OUT=${DATA}/distill_st.tsv

python ../DATA/create_distillation_tsv.py \
    --train-file ${TRAIN} \
    --distill-file ${DECODED} \
    --out-file ${OUT} \
    --verbose 