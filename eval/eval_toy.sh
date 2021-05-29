#!/usr/bin/env bash
TASK=toy_dist2
SPLIT=valid
EXP=../exp
. ${EXP}/data_path.sh
CHECKDIR=${EXP}/checkpoints/${TASK}
DATABIN=../DATA/toy_data/data-bin
SRC=chr
TGT=num
AVG=false
RESULT=./mt.results

GENARGS="--beam 5 --max-len-a 1.2 --max-len-b 10 --lenpen 1.1 --from-encoder"

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
  --path ${CHECKDIR}/${CHECKPOINT_FILENAME} --max-tokens 8000 --fp16 \
  --model-overrides '{"load_pretrained_encoder_from": None}' \
  ${GENARGS}