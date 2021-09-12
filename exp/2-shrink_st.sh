#!/usr/bin/env bash
TASK=shrink_st_test
. ./data_path.sh
CHECKPOINT=checkpoints/ctc_asr/checkpoint_best.pt

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    --load-pretrained-encoder-from ${CHECKPOINT} \
    --config-yaml config_st.yaml \
    --train-subset train_pho_st \
    --valid-subset dev_pho_st \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens 40000 \
    --update-freq 8 \
    --task speech_to_text_infer \
    --inference-config-yaml infer_st.yaml \
    --arch ws_transformer_s \
    --criterion label_smoothed_mtl --label-smoothing 0.1 --asr-factor 0.3 --report-accuracy \
    --clip-norm 10.0 \
    --optimizer adam --lr 1e-3 --lr-scheduler inverse_sqrt \
    --warmup-updates 10000 \
    --max-update 300000 \
    --save-dir checkpoints/${TASK} \
    --no-epoch-checkpoints \
    --wandb-project simulst \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-interval-updates 500 \
    --keep-interval-updates 1 \
    --keep-best-checkpoints 5 \
    --patience 50 \
    --log-format simple --log-interval 50 \
    --num-workers 4 \
    --fp16 \
    --seed 2
    