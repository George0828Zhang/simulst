#!/usr/bin/env bash
TASK=offline_asr
. ./data_path.sh

export CUDA_VISIBLE_DEVICES=0

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    --config-yaml config_asr.yaml --train-subset train_asr --valid-subset dev_asr \
    --max-tokens 20000 \
    --update-freq 8 \
    --task speech_to_text_infer  \
    --inference-config-yaml infer_asr.yaml \
    --arch s2t_transformer_s  --encoder-freezing-updates 0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --clip-norm 10.0 \
    --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
    --warmup-updates 10000 \
    --max-update 100000 \
    --tensorboard-logdir logdir/${TASK} \
    --save-dir checkpoints/${TASK} \
    --no-epoch-checkpoints \
    --best-checkpoint-metric wer \
    --save-interval-updates 500 \
    --keep-interval-updates 5 \
    --keep-best-checkpoints 5 \
    --patience 60 \
    --log-format simple --log-interval 10 \
    --num-workers 4 \
    --fp16 \
    --seed 1