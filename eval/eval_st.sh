#!/usr/bin/env bash
TASK=s2t_fixed_seq2seq_scratch
SPLIT=dev
EXP=../exp
. ${EXP}/data_path.sh
CONF=$DATA/config_st_${SRC}_${TGT}.yaml
CHECKDIR=${EXP}/checkpoints/${TASK}
RESULTS=${TASK}.${SPLIT}.results/
AVG=false
BLEU_TOK=13a

if [[ ${TGT} == "zh" ]] || [[ ${TGT} == "zh-CN" ]]; then
    BLEU_TOK=zh
fi
GENARGS="--beam 1
--max-len-a 0.1
--max-len-b 10
--post-process sentencepiece
--scoring sacrebleu
--sacrebleu-tokenizer ${BLEU_TOK}
"

if [[ $AVG == "true" ]]; then
    CHECKPOINT_FILENAME=avg_best_5_checkpoint.pt
    if [ ! -f ${CHECKDIR}/${CHECKPOINT_FILENAME} ]; then
        python ../scripts/average_checkpoints.py \
            --inputs ${CHECKDIR} --num-best-checkpoints 5 \
            --output "${CHECKDIR}/${CHECKPOINT_FILENAME}"
    fi
else
    CHECKPOINT_FILENAME=checkpoint_best.pt
fi

lang=${TGT}
mkdir -p ${RESULTS}

# python -m fairseq_cli.generate ${DATA} --user-dir ${USERDIR} \
#     --config-yaml ${CONF} --gen-subset ${SPLIT}_st_pho_${TGT} \
#     --task speech_to_text_infer \
#     --max-tokens 80000 \
#     --inference-config-yaml ../exp/infer_st.yaml \
#     --path ${CHECKDIR}/${CHECKPOINT_FILENAME} \
#     --model-overrides '{"load_pretrained_encoder_from": None}' \
#     --results-path ${RESULTS} \
#     ${GENARGS}

python ../DATA/text_processors.py zh < data/${SPLIT}.${lang} > ${RESULTS}/refs.${lang}
cat data/${SPLIT}.${lang} > ${RESULTS}/refs.${lang}
cat data/${SPLIT}.wav_list | \
    python -m fairseq_cli.interactive ${DATA} --user-dir ${USERDIR} \
    --config-yaml ${CONF} --gen-subset ${SPLIT}_st_pho_${TGT} \
    --task speech_to_text_infer \
    --buffer-size 1024 --batch-size 256 \
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
    -tok ${BLEU_TOK} | tee ${RESULTS}/score.${lang}