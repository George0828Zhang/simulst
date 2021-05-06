#!/usr/bin/env bash
TASK=wait_9
. ./data_path.sh
ASR_MODEL=./checkpoints/must_c_de_asr_transformer_s.pt

export CUDA_VISIBLE_DEVICES=0

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    --load-pretrained-encoder-from ${ASR_MODEL} \
    --config-yaml config_st.yaml --train-subset train_st --valid-subset dev_st \
    --max-tokens 20000 \
    --update-freq 8 \
    --task speech_to_text_infer  \
    --inference-config-yaml infer_simulst.yaml \
    --arch s2t_transformer_simul_trans_s --encoder-freezing-updates 0 \
    --simul-type waitk_fixed_pre_decision  \
    --waitk-lagging 9 \
    --fixed-pre-decision-ratio 7 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --clip-norm 10.0 \
    --optimizer adam --lr 1e-4 --lr-scheduler inverse_sqrt \
    --warmup-updates 10000 \
    --max-update 100000 \
    --wandb-project simulst \
    --save-dir checkpoints/${TASK} \
    --no-epoch-checkpoints \
    --save-interval-updates 200 \
    --keep-interval-updates 5 \
    --patience 50 \
    --log-format simple --log-interval 10 \
    --num-workers 4 \
    --seed 2