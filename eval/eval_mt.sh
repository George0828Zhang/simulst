#!/usr/bin/env bash
export TGT=es
TASK=mt_${TGT}
SPLIT=valid
EXP=../exp
. ${EXP}/data_path.sh
CHECKDIR=${EXP}/checkpoints/${TASK}
DATABIN=${DATA}/mt/data-bin
AVG=true
RESULT=./${TASK}.results

EXTRAARGS="--scoring sacrebleu --sacrebleu-tokenizer 13a --sacrebleu-lowercase"
GENARGS="--beam 5 --max-len-a 1.2 --max-len-b 10 \
--remove-bpe sentencepiece"

export CUDA_VISIBLE_DEVICES=0

if [[ $AVG == "true" ]]; then
  CHECKPOINT_FILENAME=avg_best_5_checkpoint.pt
  python ../scripts/average_checkpoints.py \
    --inputs ${CHECKDIR} --num-best-checkpoints 5 \
    --output "${CHECKDIR}/${CHECKPOINT_FILENAME}"
else
  CHECKPOINT_FILENAME=checkpoint_best.pt
fi

python generate.py ${DATABIN} \
  --user-dir ${USERDIR} \
  --gen-subset ${SPLIT} \
  --task translation \
  --path ${CHECKDIR}/${CHECKPOINT_FILENAME} --max-tokens 8000 --fp16 \
  --model-overrides '{"load_pretrained_encoder_from": None}' \
  --results-path ${RESULT} \
  ${GENARGS} ${EXTRAARGS}
tail -1 ${RESULT}/generate-${SPLIT}.txt

# grep -E "H-[0-9]+" ${RESULT}/generate-${SPLIT}.txt | \
#   sed 's/H-//' | sort -k 1 -n | cut -f3 | \
#   python -m sacremoses -l ${TGT} detokenize -x > ${RESULT}/hypo.${TGT}

# cat ${DATA}/data/dev/txt/dev.${TGT} | \
#   python -m sacremoses -l ${TGT} \
#     normalize -q -d -p -c > ${RESULT}/ref.${TGT}

# cat ${RESULT}/hypo.${TGT} | \
#   python -m sacrebleu \
#     --tokenize 13a \
#     --width 3 \
#     ${RESULT}/ref.${TGT}