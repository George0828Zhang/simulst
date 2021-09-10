#!/usr/bin/env bash
TASK=ctc_asr
. ./data_path.sh
DATA=${DATA_ROOT}/joint
CHECKPOINT=checkpoints/mustc_fr_asr_transformer_s.pt

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    --load-pretrained-encoder-from ${CHECKPOINT} \
    --config-yaml config_asr.yaml \
    --train-subset train_de_asr,train_es_asr,train_zh_asr \
    --valid-subset dev_de_asr,dev_es_asr,dev_zh_asr \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens 80000 \
    --update-freq 4 \
    --task speech_to_text_infer --do-asr \
    --inference-config-yaml infer_asr.yaml \
    --arch speech_encoder_s \
    --criterion label_smoothed_ctc --label-smoothing 0.1 --report-accuracy \
    --clip-norm 10.0 \
    --optimizer adam --lr 1e-3 --lr-scheduler inverse_sqrt \
    --warmup-updates 10000 \
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
    --num-workers 8 \
    --fp16 \
    --seed 1
    # --max-positions-text 1024 \
    # --max-tokens-text 8000 \