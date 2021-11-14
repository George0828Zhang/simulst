#!/usr/bin/env bash
LOOK=$1
TASK=ctc${LOOK}_fixed
. ./data_path.sh
CHECKPOINT=checkpoints/ctc_asr/avg_best_5_checkpoint.pt

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    --load-pretrained-encoder-from ${CHECKPOINT} \
    --config-yaml config_st_${SRC}_${TGT}.yaml \
    --train-subset distill_st_pho_${TGT} \
    --valid-subset dev_st_pho_${TGT} \
    --max-tokens 40000 \
    --update-freq 2 \
    --task speech_to_text_infer \
    --inference-config-yaml infer_st.yaml \
    --arch st2t_causal_encoder_s --lookahead ${LOOK} --fixed-shrink-ratio 3 \
    --criterion label_smoothed_mtl --decoder-use-ctc --label-smoothing 0.1 --asr-factor 0.5 --report-accuracy \
    --clip-norm 1.0 \
    --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
    --weight-decay 1e-4 \
    --warmup-updates 10000 \
    --max-update 300000 \
    --save-dir checkpoints/${TASK} \
    --no-epoch-checkpoints \
    --wandb-project simulst-covost \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-interval-updates 500 \
    --keep-interval-updates 1 \
    --keep-best-checkpoints 5 \
    --patience 50 \
    --log-format simple --log-interval 50 \
    --num-workers 4 \
    --fp16 \
    --seed 2