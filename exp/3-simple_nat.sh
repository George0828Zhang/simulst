#!/usr/bin/env bash
TASK=simple_nat_ucmvn
. ./data_path.sh
ASR_MODEL=./checkpoints/avg_best_5_asr_mustc_en_es.pt

export CUDA_VISIBLE_DEVICES=0

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    --load-pretrained-encoder-from ${ASR_MODEL} \
    --config-yaml config_st.yaml --train-subset train_st --valid-subset dev_st \
    --max-tokens 20000 \
    --update-freq 8 \
    --task speech_to_text_infer  \
    --inference-config-yaml infer_simulst.yaml \
    --arch simple_nat_s \
    --criterion label_smoothed_ctc --label-smoothing 0.05 \
    --clip-norm 1.0 \
    --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
    --adam-betas '(0.9, 0.98)' --adam-eps 1e-9 \
    --weight-decay 0.0001 \
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
    --seed 73

    