#!/usr/bin/env bash
TASK=toy_mt_wait1_tau0p1
SPLIT=valid
EXP=../exp
. ${EXP}/data_path.sh
CHECKDIR=${EXP}/checkpoints/${TASK}
DATABIN=${DATA}/mt/data-bin
# DATABIN=../DATA/toy_data/data-bin
# SRC=chr
# TGT=num
AVG=false

EXTRAARGS="--scoring sacrebleu --sacrebleu-tokenizer 13a --sacrebleu-lowercase"
GENARGS="--from-encoder \
--remove-bpe sentencepiece --tokenizer moses -s ${SRC} -t ${TGT} --moses-no-escape"

export CUDA_VISIBLE_DEVICES=0

if [[ $AVG == "true" ]]; then
  CHECKPOINT_FILENAME=avg_best_5_checkpoint.pt
  python ../scripts/average_checkpoints.py \
    --inputs ${CHECKDIR} --num-best-checkpoints 5 \
    --output "${CHECKDIR}/${CHECKPOINT_FILENAME}"
else
  CHECKPOINT_FILENAME=checkpoint_best.pt
fi

python -m fairseq_cli.generate ${DATABIN} \
  --user-dir ${USERDIR} \
  --gen-subset ${SPLIT} \
  --task translation_infer \
  --path ${CHECKDIR}/${CHECKPOINT_FILENAME} --max-tokens 8000 \
  --model-overrides '{"load_pretrained_encoder_from": None}' \
  ${GENARGS} ${EXTRAARGS}