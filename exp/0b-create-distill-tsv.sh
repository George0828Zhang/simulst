#!/usr/bin/env bash
. ./data_path.sh
DECODED=./distilled/generate-test.txt
TRAIN=${DATA}/train_st.tsv
OUT=${DATA}/distill_st.tsv

# grep -E "D-[0-9]+" ${DECODED} | head
python ../DATA/create_distillation_tsv.py \
    --train-file ${TRAIN} \
    --distill-file ${DECODED} \
    --out-file ${OUT}