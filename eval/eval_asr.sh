#!/usr/bin/env bash
ARCH=${1:-s2t}
TASK=${ARCH}_ctc_asr
SPLIT=${2:-dev}
EXP=../exp
. ${EXP}/data_path.sh
# DATA=${DATA_ROOT}/joint
CONF=$DATA/config_asr.yaml
CHECKDIR=${EXP}/checkpoints/${TASK}
RESULTS=${TASK}.${SPLIT}.results/
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
SRC=pho
for lang in zh-CN ; do
    echo "# evaluating lang: ${lang}"
    tsv=${DATA}/${SPLIT}_st_${SRC}_${lang}.tsv
    tail +2 ${tsv} | cut -f2 > ${RESULTS}/feats.${lang}
    tail +2 ${tsv} | cut -f4 > ${RESULTS}/refs.${lang}

    cat ${RESULTS}/feats.${lang} | \
    python -m fairseq_cli.interactive ${DATA} --user-dir ${USERDIR} \
        --config-yaml ${CONF} \
        --gen-subset ${SPLIT}_st_${SRC}_${TGT} \
        --task speech_to_text_infer --do-asr \
        --buffer-size 128 --batch-size 128 \
        --inference-config-yaml infer_asr.yaml \
        --path ${CHECKDIR}/${CHECKPOINT_FILENAME} \
        --model-overrides '{"load_pretrained_encoder_from": None}' \
        ${EXTRAARGS} | \
    grep -E "H-[0-9]+" | \
    cut -f3 > ${RESULTS}/hyps.${lang}
    wer ${RESULTS}/refs.${lang} ${RESULTS}/hyps.${lang} | tee ${RESULTS}/score.${lang}
done
