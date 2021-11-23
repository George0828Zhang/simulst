#!/usr/bin/env bash
ARCH=${1:-s2t}
TASK=${ARCH}_ctc_asr
. ./data_path.sh

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    --config-yaml config_asr.yaml \
    --train-subset train_st_pho_${TGT} \
    --valid-subset dev_st_pho_${TGT} \
    --max-tokens 160000 \
    --update-freq 2 \
    --task speech_to_text_infer --do-asr \
    --inference-config-yaml infer_asr.yaml \
    --arch ${ARCH}_speech_encoder_s \
    --criterion label_smoothed_ctc --label-smoothing 0.1 --report-accuracy \
    --clip-norm 15.0 \
    --optimizer adam --lr 5e-4 --lr-scheduler inverse_sqrt \
    --warmup-updates 10000 \
    --max-update 300000 \
    --save-dir checkpoints/${TASK} \
    --no-epoch-checkpoints \
    --wandb-project simulst-covost \
    --best-checkpoint-metric wer \
    --save-interval-updates 500 \
    --keep-interval-updates 1 \
    --keep-best-checkpoints 5 \
    --patience 50 \
    --log-format simple --log-interval 50 \
    --num-workers 4 \
    --fp16 \
    --seed 1
