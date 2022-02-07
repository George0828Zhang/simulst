#!/usr/bin/env bash
TASK=test_ssnt_asr
SPLIT=dev
EXP=../exp
. ${EXP}/data_path.sh
CONF=$DATA/config_st.yaml
CHECKDIR=${EXP}/checkpoints/${TASK}
RESULTS=asr.${SPLIT}.results/
AVG=false

EXTRAARGS=""

if [[ $AVG == "true" ]]; then
    CHECKPOINT_FILENAME=avg_best_5_checkpoint.pt
    python ../scripts/average_checkpoints.py \
      --inputs ${CHECKDIR} --num-best-checkpoints 5 \
      --output "${CHECKDIR}/${CHECKPOINT_FILENAME}"
else
    CHECKPOINT_FILENAME=checkpoint_best.pt
fi

mkdir -p ${RESULTS}
python generate.py ${DATA} --user-dir ${USERDIR} \
        --config-yaml ${CONF} \
        --gen-subset ${SPLIT}_st \
        --task speech_to_text_infer --do-asr \
        --inference-config-yaml infer_asr.yaml \
        --path ${CHECKDIR}/${CHECKPOINT_FILENAME} \
        --model-overrides '{"load_pretrained_encoder_from": None}' \
        ${EXTRAARGS}