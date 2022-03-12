#!/usr/bin/env bash
export TGT=de
TASK=ctc_s2s_asr_${TGT}
SPLIT=dev
EXP=$(realpath ../exp)
. ${EXP}/data_path.sh
CONF=$DATA/config_st.yaml
CHECKDIR=${EXP}/checkpoints/${TASK}
RESULTS=asr_${TGT}.${SPLIT}.results
AVG=true

EXTRAARGS="--scoring wer --wer-remove-punct --wer-lowercase --wer-tokenizer 13a --beam 5 --max-len-a 0.1 --max-len-b 10"

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
        --results-path ${RESULTS} \
        --path ${CHECKDIR}/${CHECKPOINT_FILENAME} \
        --model-overrides '{"load_pretrained_encoder_from": None}' \
        ${EXTRAARGS}
tail -1 ${RESULTS}/generate-${SPLIT}_st.txt
