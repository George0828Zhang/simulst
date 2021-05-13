#!/usr/bin/env bash
TASK=offline_asr
. ./data_path.sh

export CUDA_VISIBLE_DEVICES=0

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    --config-yaml config_asr.yaml --train-subset train_asr --valid-subset dev_asr \
    --max-tokens 40000 \
    --update-freq 8 \
    --task speech_to_text_infer  \
    --inference-config-yaml infer_asr.yaml \
    --arch waitk_s2t_transformer_s \
    --waitk 60000 --pre-decision-ratio 1 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --clip-norm 10.0 \
    --optimizer adam --lr 1e-3 --lr-scheduler inverse_sqrt \
    --warmup-updates 10000 \
    --max-update 300000 \
    --wandb-project simulst \
    --save-dir checkpoints/${TASK} \
    --no-epoch-checkpoints \
    --best-checkpoint-metric wer \
    --save-interval-updates 500 \
    --keep-interval-updates 5 \
    --keep-best-checkpoints 5 \
    --patience 75 \
    --log-format simple --log-interval 50 \
    --num-workers 8 \
    --fp16 \
    --seed 16