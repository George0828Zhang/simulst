#!/usr/bin/env bash
TASK=sinkhorn_nat
. ./data_path.sh
ASR_MODEL=./checkpoints/fb_offline_asr/avg_best_5_checkpoint.pt

export CUDA_VISIBLE_DEVICES=0

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    --load-pretrained-encoder-from ${ASR_MODEL} \
    --config-yaml config_st.yaml --train-subset train_st --valid-subset dev_st \
    --max-tokens 20000 \
    --update-freq 8 \
    --task speech_to_text_infer  \
    --inference-config-yaml infer_simulst.yaml \
    --arch sinkhorn_nat_s \
    --sinkhorn-iters 8 --sinkhorn-tau 0.75 \
    --criterion label_smoothed_ctc --label-smoothing 0.1 \
    --clip-norm 10.0 \
    --optimizer adam --lr 1e-3 --lr-scheduler inverse_sqrt \
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
    --fp16 \
    --seed 2

    