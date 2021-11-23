#!/usr/bin/env bash
ARCH=${1:-s2t}
TASK=${ARCH}_ws_seq2seq
. ./data_path.sh
CHECKASR=checkpoints/${ARCH}_ctc_asr/avg_best_5_checkpoint.pt

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    --load-pretrained-encoder-from ${CHECKASR} \
    --config-yaml config_st_${SRC}_${TGT}.yaml \
    --train-subset distill_st_pho_${TGT} \
    --valid-subset dev_st_pho_${TGT} \
    --max-tokens 80000 \
    --update-freq 4 \
    --task speech_to_text_infer \
    --inference-config-yaml infer_st.yaml \
    --arch ${ARCH}_seq2seq_s --do-weighted-shrink \
    --criterion label_smoothed_mtl --label-smoothing 0.1 --asr-factor 0.3 --report-accuracy \
    --clip-norm 1.0 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr 2e-3 --lr-scheduler inverse_sqrt \
    --dropout 0.15 --warmup-init-lr 1e-7 --weight-decay 0.01 \
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
    
