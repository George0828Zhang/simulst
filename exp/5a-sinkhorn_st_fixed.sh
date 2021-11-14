#!/usr/bin/env bash
DELAY=1
TASK=sinkhorn_st_fixed_ft_no_ctx
. ./data_path.sh
CHECKPOINT=checkpoints/causal_st_fixed/checkpoint_best.pt
# TODO: mask scheduling
python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    --load-pretrained-cascade-from ${CHECKPOINT} \
    --load-pretrained-decoder-from ${CHECKPOINT} \
    --config-yaml config_st_${SRC}_${TGT}.yaml \
    --train-subset distill_st_pho_${TGT} \
    --valid-subset dev_st_pho_${TGT} \
    --max-tokens 40000 \
    --update-freq 2 \
    --task speech_to_text_infer \
    --inference-config-yaml infer_st.yaml \
    --arch st2t_sinkhorn_encoder_s --fixed-shrink-ratio 3 --non-causal-layers 3 \
    --mask-ratio 1.0 --sinkhorn-iters 16 --sinkhorn-tau 0.25 --sinkhorn-noise-factor 0.3 --sinkhorn-bucket-size 1 --sinkhorn-energy dot \
    --criterion label_smoothed_mtl --decoder-use-ctc --label-smoothing 0.1 --asr-factor 0.5 --report-accuracy --report-sinkhorn-dist \
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
