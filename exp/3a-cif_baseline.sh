#!/usr/bin/env bash
export TGT=de
LATENCY=${1:-0.0}
TASK=cif_base_${LATENCY//./_}
. ./data_path.sh
ASR_CHECK=checkpoints/ctc_s2s_asr/avg_best_5_checkpoint.pt

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
    --arch cif_transformer_s --share-decoder-input-output-embed --cif-sg-alpha \
    --dropout 0.3 --activation-dropout 0.1 --attention-dropout 0.1 \
    --criterion cif_loss --label-smoothing 0.1 --quant-type sum --ctc-factor 0.0 --latency-factor ${LAT} \
    --clip-norm 10 --weight-decay 1e-4 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr 5e-4 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 --warmup-init-lr 1e-7 \
    --max-update 200000 \
    --save-dir checkpoints/${TASK} \
    --wandb-project simulst-cif-final \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --keep-last-epochs 1 \
    --keep-best-checkpoints 5 \
    --patience 25 \
    --log-format simple --log-interval 50 \
    --num-workers ${WORKERS} \
    --fp16 --fp16-init-scale 1 --memory-efficient-fp16 \
    --seed 999
