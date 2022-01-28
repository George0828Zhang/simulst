#!/usr/bin/env bash
TASK=mt
. ./data_path.sh
DATA=${DATA}/mt/data-bin

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    -s ${SRC} -t ${TGT} \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens 8192 \
    --update-freq 8 \
    --task translation_infer \
    --inference-config-yaml infer_mt.yaml \
    --arch transformer_small --share-all-embeddings \
    --dropout 0.3 --activation-dropout 0.1 --attention-dropout 0.1 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
    --clip-norm 2 --weight-decay 1e-4 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr 5e-4 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 --warmup-init-lr 1e-7 \
    --max-update 300000 \
    --save-dir checkpoints/${TASK} \
    --wandb-project simulst \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-interval 4 \
    --keep-last-epochs 1 \
    --keep-best-checkpoints 5 \
    --patience 25 \
    --log-format simple --log-interval 50 \
    --num-workers 4 \
    --fp16 \
    --seed 1