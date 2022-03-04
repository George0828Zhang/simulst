#!/usr/bin/env bash
TASK=ctc_s2s_asr
SPLIT=tst-COMMON
EXP=$(realpath ../exp)
. ${EXP}/data_path.sh
CONF=$DATA/config_st.yaml
INF=${EXP}/infer_asr.yaml
CHECKDIR=${EXP}/checkpoints/${TASK}
RESULTS=asr.${SPLIT}.results/
AVG=true

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

tsv=${DATA}/${SPLIT}_st.tsv
tail +2 ${tsv} | cut -f2 > ${RESULTS}/feats.${TGT}
tail +2 ${tsv} | cut -f4 > ${RESULTS}/refs.${TGT}
cat ${RESULTS}/feats.${TGT} | \
    python interactive.py ${DATA} --user-dir ${USERDIR} \
        --config-yaml ${CONF} \
        --gen-subset ${SPLIT}_st \
        --task speech_to_text_infer --do-asr \
        --buffer-size 1500 --batch-size 32 \
        --inference-config-yaml ${INF} \
        --path ${CHECKDIR}/${CHECKPOINT_FILENAME} \
        ${EXTRAARGS} | \
    grep -E "D-[0-9]+" | \
    cut -f3 > ${RESULTS}/hyps.${TGT}
script=$(which wer)
python ${script} ${RESULTS}/refs.${TGT} ${RESULTS}/hyps.${TGT} | tee ${RESULTS}/interactive_score.txt
