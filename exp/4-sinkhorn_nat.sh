#!/usr/bin/env bash
TASK=st_sort_ctc
. ./data_path.sh
ASR_MODEL=./checkpoints/fb_asr.pt

export CUDA_VISIBLE_DEVICES=0

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    --load-pretrained-encoder-from ${ASR_MODEL} \
    --config-yaml config_st.yaml --train-subset train_distill --valid-subset dev_st \
    --max-tokens 20000 \
    --update-freq 8 \
    --task speech_to_text_infer  \
    --inference-config-yaml infer_simulst.yaml \
    --arch sinkhorn_encoder_s \
    --sinkhorn-iters 8 --sinkhorn-tau 0.75 --sinkhorn-noise-factor 0.1 --sinkhorn-bucket-size 1 --sinkhorn-energy l2 \
    --criterion label_smoothed_ctc --label-smoothing 0.1 --report-sinkhorn-dist --report-accuracy --decoder-use-ctc \
    --clip-norm 10.0 \
    --weight-decay 0.0001 \
    --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
    --warmup-updates 10000 \
    --max-update 50000 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --wandb-project sinkhorn \
    --save-dir checkpoints/${TASK} \
    --no-epoch-checkpoints \
    --save-interval-updates 500 \
    --keep-interval-updates 5 \
    --keep-best-checkpoints 5 \
    --patience 50 \
    --log-format simple --log-interval 10 \
    --num-workers 8 \
    --seed 2 \
    --fp16