#!/usr/bin/env bash
TASK=ws_mma
. ./data_path.sh
CHECKPOINT=checkpoints/ctc_asr/avg_best_5_checkpoint.pt

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    --load-pretrained-encoder-from ${CHECKPOINT} \
    --config-yaml config_st.yaml \
    --train-subset train_pho_st \
    --valid-subset dev_pho_st \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens 40000 \
    --max-tokens-valid 10000 \
    --update-freq 2 \
    --task speech_to_text_infer \
    --arch ws_transformer_monotonic_s --do-weighted-shrink \
    --simul-type hard_aligned --mass-preservation \
    --criterion label_smoothed_mtl --label-smoothing 0.1 --asr-factor 0.5 --report-accuracy \
    --clip-norm 1.0 \
    --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
    --weight-decay 1e-4 \
    --warmup-updates 10000 \
    --max-update 300000 \
    --save-dir checkpoints/${TASK} \
    --no-epoch-checkpoints \
    --wandb-project simulst \
    --best-checkpoint-metric f1 --maximize-best-checkpoint-metric \
    --save-interval-updates 500 \
    --keep-interval-updates 1 \
    --keep-best-checkpoints 5 \
    --patience 50 \
    --log-format simple --log-interval 50 \
    --num-workers 4 \
    --fp16 \
    --seed 2
    