#!/usr/bin/env bash
WAITK=9
TASK=wait_${WAITK}_sinkhorn
. ./data_path.sh
ASR_MODEL=./checkpoints/mustc_de_asr_transformer_s_causal.pt

export CUDA_VISIBLE_DEVICES=1

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    --load-pretrained-encoder-from ${ASR_MODEL} \
    --config-yaml config_st.yaml --train-subset train_st --valid-subset dev_st \
    --max-tokens 20000 \
    --update-freq 8 \
    --task speech_to_text_infer  \
    --inference-config-yaml infer_simulst.yaml \
    --arch waitk_s2t_transformer_s \
    --waitk ${WAITK} --pre-decision-ratio 7 \
    --criterion label_smoothed_cross_entropy_sinkhorn --label-smoothing 0.1 \
    --aux-factor 1.0 --aux-type dot --stop-grad-embeddings --sinkhorn-temperature 0.7 --sinkhorn-iters 8 \
    --clip-norm 10.0 \
    --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
    --warmup-updates 10000 \
    --max-update 50000 \
    --wandb-project simulst \
    --save-dir checkpoints/${TASK} \
    --no-epoch-checkpoints \
    --save-interval-updates 500 \
    --keep-interval-updates 5 \
    --keep-best-checkpoints 5 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --patience 50 \
    --log-format simple --log-interval 10 \
    --num-workers 8 \
    --seed 2