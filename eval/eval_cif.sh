#!/usr/bin/env bash
# TASK=cif_de_align_ctc0_3_lat0_0
    # cif_de_align_ctc0_3_lat1_0
    # cif_de_il_sum_ctc0_3_lat0_0
    # cif_de_il_sum_ctc0_3_lat1_0
TASK=cif_de_il_sum_ctc0_3_lat0_5
SPLIT=dev
EXP=../exp
. ${EXP}/data_path.sh
CONF=$DATA/config_st.yaml
CHECKDIR=${EXP}/checkpoints/${TASK}
RESULTS=${TASK}.${SPLIT}.results
AVG=true

EXTRAARGS="--scoring sacrebleu --beam 1 --max-len-a 0.1 --max-len-b 10"

if [[ $AVG == "true" ]]; then
    CHECKPOINT_FILENAME=avg_best_5_checkpoint.pt
    python ../scripts/average_checkpoints.py \
      --inputs ${CHECKDIR} --num-best-checkpoints 5 \
      --output "${CHECKDIR}/${CHECKPOINT_FILENAME}"
else
    CHECKPOINT_FILENAME=checkpoint_last.pt
fi

mkdir -p ${RESULTS}

tsv=${DATA}/${SPLIT}_st.tsv
tail +2 ${tsv} | cut -f2 > ${RESULTS}/feats.${TGT}
tail +2 ${tsv} | cut -f5 > ${RESULTS}/refs.${TGT}
# # data/dev.wav_list
# # data/dev.${TGT}
# cat ${RESULTS}/feats.${TGT} | \
#     python interactive.py ${DATA} --user-dir ${USERDIR} \
#         --config-yaml ${CONF} \
#         --gen-subset ${SPLIT}_st \
#         --task speech_to_text_infer \
#         --buffer-size 1500 --batch-size 32 \
#         --inference-config-yaml infer_mt.yaml \
#         --path ${CHECKDIR}/${CHECKPOINT_FILENAME} \
#         --model-overrides '{"load_pretrained_encoder_from": None}' \
#         --post-process sentencepiece \
#         ${EXTRAARGS} | \
#     grep -E "H-[0-9]+" | \
#     cut -f3 > ${RESULTS}/hyps.${TGT}
# python -m sacrebleu ${RESULTS}/refs.${TGT} \
#     -i ${RESULTS}/hyps.${TGT} \
#     --width 2 \
#     --tok 13a > ${RESULTS}/interactive_score.txt

python generate.py ${DATA} --user-dir ${USERDIR} \
        --config-yaml ${CONF} \
        --gen-subset ${SPLIT}_st \
        --task speech_to_text_infer \
        --path ${CHECKDIR}/${CHECKPOINT_FILENAME} \
        --model-overrides '{"load_pretrained_encoder_from": None, "cif_beta": 1.0}' \
        --results-path ${RESULTS} \
        ${EXTRAARGS}

grep -E "D-[0-9]+" ${RESULTS}/generate-${SPLIT}_st.txt | \
    sed s/^D-// | sort -n -k 1 | \
    cut -f3 > ${RESULTS}/hyps-gen.${TGT}
python -m sacrebleu ${RESULTS}/refs.${TGT} \
    -i ${RESULTS}/hyps-gen.${TGT} \
    --width 2 \
    --tok 13a > ${RESULTS}/generate_score.txt
cat ${RESULTS}/generate_score.txt