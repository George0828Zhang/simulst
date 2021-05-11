#!/usr/bin/env bash
TASK=offline_mt
. ./data_path.sh
DATA=${DATA}/mt/data-bin
export CUDA_VISIBLE_DEVICES=0

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    -s ${SRC} -t ${TGT} \
    --max-tokens 16000 \
    --update-freq 1 \
    --task translation \
    --arch transformer \
    --encoder-embed-dim 256 --decoder-embed-dim 256 \
    --encoder-ffn-embed-dim 2048 --decoder-ffn-embed-dim 2048 \
    --encoder-attention-heads 4 --decoder-attention-heads 4 \
    --encoder-normalize-before --decoder-normalize-before \
    --share-decoder-input-output-embed \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --clip-norm 10.0 \
    --weight-decay 0.0001 \
    --optimizer adam --lr 0.0005 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --max-update 50000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-remove-bpe sentencepiece \
    --eval-bleu-detok moses --eval-bleu-detok-args '{"target_lang": "de", "moses_no_escape": true}' \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --wandb-project simulst \
    --save-dir checkpoints/${TASK} \
    --no-epoch-checkpoints \
    --save-interval-updates 500 \
    --keep-interval-updates 5 \
    --keep-best-checkpoints 5 \
    --patience 50 \
    --log-format simple --log-interval 50 \
    --num-workers 8 \
    --seed 1 \
    --fp16