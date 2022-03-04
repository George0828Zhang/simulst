#!/usr/bin/env bash
LATENCY=${1:-0.0}
TASK=mma_${LATENCY//./_}
. ./data_path.sh
ASR_CHECK=checkpoints/test_ctc_asr_load/checkpoint_best.pt

echo "Train MMA with latency weight: ${LATENCY}"
python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    --load-pretrained-encoder-from ${ASR_CHECK} \
    --config-yaml config_st.yaml \
    --train-subset distill_st,train_st \
    --valid-subset dev_st \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens 20000 \
    --update-freq 8 \
    --task speech_to_text_infer \
    --inference-config-yaml infer_st.yaml \
    --arch mma_model_s --share-decoder-input-output-embed \
    --simul-attn-type infinite_lookback_fixed_pre_decision \
    --fixed-pre-decision-type last --fixed-pre-decision-ratio 8 \
    --energy-bias --mass-preservation \
    --dropout 0.3 --activation-dropout 0.1 --attention-dropout 0.1 \
    --criterion mma_criterion --label-smoothing 0.1 --latency-avg-weight ${LATENCY} --latency-var-weight ${LATENCY} \
    --clip-norm 10 --weight-decay 1e-4 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr 5e-4 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 --warmup-init-lr 1e-7 \
    --max-update 200000 \
    --validate-after-updates 3000 \
    --save-dir checkpoints/${TASK} \
    --wandb-project simulst-cif \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --keep-last-epochs 1 \
    --keep-best-checkpoints 5 \
    --patience 25 \
    --log-format simple --log-interval 50 \
    --num-workers ${WORKERS} \
    --fp16 --fp16-init-scale 1 --memory-efficient-fp16 \
    --seed 1
