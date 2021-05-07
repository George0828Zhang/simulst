#!/usr/bin/env bash
. ./data_path.sh
DECODED=./distilled/generate-train_${lang}.txt
TRAIN=${DATA}/train_${lang}.tsv
OUT=${DATA}/distill_${lang}.tsv

# grep -E "D-[0-9]+" ${DECODED} | head
python ../DATA/create_distillation_tsv.py \
    --train-file ${TRAIN} \
    --distill-file ${DECODED} \
    --out-file ${OUT}