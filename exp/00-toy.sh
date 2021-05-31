#!/usr/bin/env bash
TASK=toy_attn_sort_ctc
. ./data_path.sh
DATA=../DATA/toy_data/data-bin
SRC=chr
TGT=num
export CUDA_VISIBLE_DEVICES=0

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    -s ${SRC} -t ${TGT} \
    --max-tokens 8000 \
    --update-freq 2 \
    --task translation_infer \
    --inference-config-yaml infer_toy.yaml \
    --arch toy_transformer \
    --fusion-factor 0.0 --fusion-type inter \
    --sinkhorn-iters 8 --sinkhorn-tau 0.75 --sinkhorn-noise-factor 0.1 --sinkhorn-bucket-size 1 --sinkhorn-energy l2 \
    --criterion label_smoothed_ctc --label-smoothing 0.1 --report-sinkhorn-dist --report-accuracy --decoder-use-ctc \
    --clip-norm 10.0 \
    --weight-decay 0.0001 \
    --optimizer adam --lr 0.0005 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --max-update 50000 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --wandb-project sinkhorn \
    --save-dir checkpoints/${TASK} \
    --save-interval-updates 1000 \
    --keep-interval-updates 5 \
    --keep-last-epochs 1 \
    --keep-best-checkpoints 1 \
    --patience 50 \
    --log-format simple --log-interval 50 \
    --num-workers 8 \
    --seed 1 \
    --fp16