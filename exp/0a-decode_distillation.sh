#!/usr/bin/env bash
TASK=mt
. ./data_path.sh
CHECKDIR=./checkpoints/${TASK}
DATABIN=${DATA}/mt/data-bin
CHECKPOINT_FILENAME=avg_best_5_checkpoint.pt
RESULT=./distilled

EXTRAARGS="--scoring sacrebleu --sacrebleu-tokenizer zh"
GENARGS="--beam 5 --max-len-a 1.2 --max-len-b 10 --lenpen 1 --remove-bpe sentencepiece"

export CUDA_VISIBLE_DEVICES=0

python -m fairseq_cli.generate ${DATABIN} \
  -s ${SRC} -t ${TGT} \
  --user-dir ${USERDIR} \
  --gen-subset train \
  --task translation \
  --path ${CHECKDIR}/${CHECKPOINT_FILENAME} --max-tokens 32000 --fp16 \
  --model-overrides '{"load_pretrained_encoder_from": None}' \
  --results-path ${RESULT} \
  ${GENARGS} ${EXTRAARGS}