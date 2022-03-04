#!/usr/bin/env bash
TASK=ctc_s2s_asr
. ./data_path.sh

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    --config-yaml config_st.yaml \
    --train-subset train_st \
    --valid-subset dev_st \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens 40000 \
    --update-freq 4 \
    --task speech_to_text_infer --do-asr \
    --inference-config-yaml infer_asr.yaml \
    --arch s2t_emformer_s --ctc-layer \
    --share-decoder-input-output-embed \
    --dropout 0.3 --activation-dropout 0.1 --attention-dropout 0.1 \
    --criterion joint_ctc_criterion --label-smoothing 0.1 --report-accuracy \
    --clip-norm 10 --weight-decay 1e-4 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr 5e-4 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 --warmup-init-lr 1e-7 \
    --max-update 300000 \
    --save-dir checkpoints/${TASK} \
    --wandb-project simulst-cif-final \
    --best-checkpoint-metric wer \
    --validate-after-updates 10000 \
    --validate-interval 2 \
    --save-interval 1 \
    --keep-last-epochs 1 \
    --keep-best-checkpoints 5 \
    --patience 25 \
    --log-format simple --log-interval 50 \
    --num-workers 4 \
    --fp16 --fp16-init-scale 1 --memory-efficient-fp16 \
    --seed 999