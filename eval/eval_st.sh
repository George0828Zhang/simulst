#!/usr/bin/env bash
TASK=fixed_waitk_9
SPLIT=dev
EXP=../exp
. ${EXP}/data_path.sh
CONF=$DATA/config_st.yaml
CHECKDIR=${EXP}/checkpoints/${TASK}
RESULTS=${TASK}.${SPLIT}.results/
AVG=true

EXTRAARGS="--scoring sacrebleu --sacrebleu-tokenizer 13a --sacrebleu-lowercase"
GENARGS="--beam 1 --remove-bpe sentencepiece"

export CUDA_VISIBLE_DEVICES=0

if [[ $AVG == "true" ]]; then
    CHECKPOINT_FILENAME=avg_best_5_checkpoint.pt
    # python ../scripts/average_checkpoints.py \
    #     --inputs ${CHECKDIR} --num-best-checkpoints 5 \
    #     --output "${CHECKDIR}/${CHECKPOINT_FILENAME}"
else
    CHECKPOINT_FILENAME=checkpoint_best.pt
fi

lang=${TGT}
mkdir -p ${RESULTS}
# tsv=${DATA}/${SPLIT}_st.tsv
# tail +2 ${tsv} | cut -f2 > ${RESULTS}/feats.${lang}
# tail +2 ${tsv} | cut -f5 > ${RESULTS}/refs.${lang}
# cat ${RESULTS}/feats.${lang} | \
cp data/${SPLIT}.${lang} ${RESULTS}/refs.${lang}
cat data/${SPLIT}.wav_list | \
    python -m fairseq_cli.interactive ${DATA} --user-dir ${USERDIR} \
    --config-yaml ${CONF} --gen-subset ${SPLIT}_st \
    --task speech_to_text_infer \
    --buffer-size 128 --batch-size 128 \
    --inference-config-yaml ../exp/infer_st.yaml \
    --path ${CHECKDIR}/${CHECKPOINT_FILENAME} \
    --model-overrides '{"load_pretrained_encoder_from": None}' \
    ${GENARGS} ${EXTRAARGS} | \
    grep -E "H-[0-9]+" | \
    cut -f3 > ${RESULTS}/hyps.${lang}

python -m sacrebleu ${RESULTS}/refs.${lang} \
    -i ${RESULTS}/hyps.${lang} \
    -m bleu \
    --width 2 \
    --tok 13a -lc | tee ${RESULTS}/score.${lang}