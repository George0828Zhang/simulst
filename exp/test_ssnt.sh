#!/usr/bin/env bash
TASK=test_ssnt_asr
. ./data_path.sh
ASR_CHECK=checkpoints/seq2seq_asr/checkpoint_best.pt

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    --load-pretrained-encoder-from ${ASR_CHECK} \
    --config-yaml config_st.yaml \
    --train-subset train_st \
    --valid-subset dev_st \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens 16384 \
    --update-freq 4 \
    --task speech_to_text_infer --do-asr \
    --inference-config-yaml infer_asr.yaml \
    --arch ssnt_model_s --share-decoder-input-output-embed \
    --dropout 0.3 --activation-dropout 0.1 --attention-dropout 0.1 \
    --criterion ssnt_criterion --memory-efficient \
    --clip-norm 2 --weight-decay 1e-4 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr 5e-4 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 --warmup-init-lr 1e-7 \
    --max-update 300000 \
    --save-dir checkpoints/${TASK} \
    --wandb-project simulst \
    --best-checkpoint-metric wer \
    --keep-last-epochs 1 \
    --keep-best-checkpoints 5 \
    --patience 25 \
    --log-format simple --log-interval 50 \
    --num-workers 4 \
    --fp16 \
    --seed 1
    # --wandb-project simulst \
    # --validate-interval 4 \
    # --save-interval 4 \
    # --validate-after-updates 10000 \
    # --wandb-project simulst \
    # --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \