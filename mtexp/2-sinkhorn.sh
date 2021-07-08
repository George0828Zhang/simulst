#!/usr/bin/env bash
TASK=sinkhorn_tau0p1
. ./data_path.sh
#DATA=${DATA}/mt/data-bin
export CUDA_VISIBLE_DEVICES=0

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    -s ${SRC} -t ${TGT} \
    --train-subset train_distill \
    --max-tokens 8000 \
    --update-freq 2 \
    --task translation_infer \
    --inference-config-yaml infer_mt.yaml \
    --arch toy_transformer_mt \
    --sinkhorn-iters 8 --sinkhorn-tau 0.1 --sinkhorn-noise-factor 0.1 --sinkhorn-bucket-size 1 --sinkhorn-energy dot \
    --criterion label_smoothed_ctc --label-smoothing 0.1 --report-sinkhorn-dist --report-accuracy --decoder-use-ctc \
    --clip-norm 10.0 \
    --weight-decay 0.0001 \
    --optimizer adam --lr 0.0005 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --max-update 50000 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --wandb-project sinkhorn \
    --save-dir checkpoints/${TASK} \
    --no-epoch-checkpoints \
    --save-interval-updates 500 \
    --keep-interval-updates 1 \
    --keep-best-checkpoints 1 \
    --patience 50 \
    --log-format simple --log-interval 50 \
    --num-workers 8 \
    --seed 1 \
    --fp16
