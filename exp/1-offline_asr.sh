#!/usr/bin/env bash
TASK=offline_asr
. ./data_path.sh

export CUDA_VISIBLE_DEVICES=0

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    --config-yaml config_st.yaml --train-subset train_asr --valid-subset dev_asr \
    --max-tokens 40000 \
    --update-freq 8 \
    --task speech_to_text  \
    --arch convtransformer_espnet  \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
    --weight-decay 0.0001 \
    --dropout 0.1 \
    --clip-norm 10.0 \
    --optimizer adam --lr 5e-4 --lr-scheduler inverse_sqrt \
    --warmup-updates 10000 \
    --max-update 100000 \
    --tensorboard-logdir logdir/${TASK} \
    --save-dir checkpoints/${TASK} \
    --no-epoch-checkpoints \
    --save-interval-updates 200 \
    --keep-interval-updates 5 \
    --patience 50 \
    --log-format simple --log-interval 10 \
    --num-workers 4 \
    --fp16 \
    --seed 66