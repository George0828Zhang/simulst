#!/usr/bin/env bash
TASK=wait_9_mt
. ./data_path.sh
DATA=${DATA}/mt/data-bin
export CUDA_VISIBLE_DEVICES=0

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    -s ${SRC} -t ${TGT} \
    --max-tokens 8000 \
    --update-freq 2 \
    --task translation  \
    --arch transformer_monotonic \
    --simul-type waitk  \
    --waitk-lagging 9 \
    --encoder-embed-dim 256 --decoder-embed-dim 256 \
    --encoder-ffn-embed-dim 2048 --decoder-ffn-embed-dim 2048 \
    --encoder-attention-heads 4 --decoder-attention-heads 4 \
    --encoder-normalize-before --decoder-normalize-before \
    --share-all-embeddings \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --clip-norm 10.0 \
    --optimizer adam --lr 5e-4 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --max-update 100000 \
    --wandb-project simulst \
    --save-dir checkpoints/${TASK} \
    --no-epoch-checkpoints \
    --save-interval-updates 200 \
    --keep-interval-updates 5 \
    --patience 50 \
    --log-format simple --log-interval 10 \
    --num-workers 4 \
    --seed 2 \
    --fp16