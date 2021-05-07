#!/usr/bin/env bash
TASK=wait_9
. ./data_path.sh
ASR_MODEL=./checkpoints/mustc_de_asr_transformer_s_causal.pt

export CUDA_VISIBLE_DEVICES=0

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    --load-pretrained-encoder-from ${ASR_MODEL} \
    --config-yaml config_st.yaml --train-subset train_st --valid-subset dev_st \
    --max-tokens 20000 \
    --update-freq 8 \
    --task speech_to_text_infer  \
    --inference-config-yaml infer_simulst.yaml \
    --arch waitk_s2t_transformer_s --encoder-freezing-updates 0 \
    --causal --waitk 9 --pre-decision-ratio 7 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --clip-norm 10.0 \
    --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
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
    --seed 2