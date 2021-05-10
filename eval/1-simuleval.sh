#!/usr/bin/env bash
TASK=wait_9
SPLIT=dev #tst-COMMON
EVAL_DATA=./data
AGENT=./agents/waitk_fixed_predecision_agent.py
EXP=../exp
. ${EXP}/data_path.sh
CONF=$DATA/config_st.yaml
CHECKDIR=${EXP}/checkpoints/${TASK}
AVG=true

export CUDA_VISIBLE_DEVICES=0

if [[ $AVG == "true" ]]; then
  CHECKPOINT_FILENAME=avg_best_5_checkpoint.pt
  python ../scripts/average_checkpoints.py \
    --inputs ${CHECKDIR} --num-best-checkpoints 5 \
    --output "${CHECKDIR}/${CHECKPOINT_FILENAME}"
else
  CHECKPOINT_FILENAME=checkpoint_best.pt
fi

SRC_LIST=${EVAL_DATA}/${SPLIT}.wav_list
TGT_FILE=${EVAL_DATA}/${SPLIT}.${TGT}
OUTPUT=${TASK}.en-${TGT}.results
simuleval \
  --agent ${AGENT} \
  --user-dir ${USERDIR} \
  --source ${SRC_LIST} \
  --target ${TGT_FILE} \
  --data-bin ${DATA} \
  --config ${CONF} \
  --model-path ${CHECKDIR}/${CHECKPOINT_FILENAME} \
  --output ${OUTPUT} \
  --scores \
  --gpu \
  --test-waitk 9