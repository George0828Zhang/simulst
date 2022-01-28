#!/usr/bin/env bash
TASK=seq2seq_asr
. ./data_path.sh

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    --config-yaml config_st.yaml \
    --train-subset train_st \
    --valid-subset dev_st \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens 80000 \
    --update-freq 4 \
    --task speech_to_text_infer --do-asr \
    --inference-config-yaml infer_asr.yaml \
    --arch s2t_transformer_convpos_s --share-decoder-input-output-embed \
    --dropout 0.3 --activation-dropout 0.1 --attention-dropout 0.1 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
    --clip-norm 2 --weight-decay 1e-4 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr 5e-4 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 --warmup-init-lr 1e-7 \
    --max-update 300000 \
    --save-dir checkpoints/${TASK} \
    --no-epoch-checkpoints \
    --wandb-project simulst \
    --best-checkpoint-metric wer \
    --save-interval-updates 500 \
    --keep-interval-updates 1 \
    --keep-best-checkpoints 5 \
    --patience 50 \
    --log-format simple --log-interval 50 \
    --num-workers 4 \
    --fp16 \
    --seed 1