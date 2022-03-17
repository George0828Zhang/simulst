#!/usr/bin/env bash
# credits: https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -t|--tgt-lang)
      TGT="$2"
      shift # past argument
      shift # past value
      ;;
    -l|--latency)
      LATVAR="$2"
      shift # past argument
      shift # past value
      ;;
    -m|--model)
      MODEL="$2"
      shift # past argument
      shift # past value
      ;;
    *)    # unknown option
      POSITIONAL+=("$1") # save it in an array for later
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL[@]}" # restore positional parameters

# defaults
export TGT=${TGT:-de}
MODEL=${MODEL:-infinite_lookback}
LATVAR=${LATVAR:-0.0}
if [[ ${MODEL} == "infinite_lookback" ]]; then
    LATAVG=${LATVAR}
else
    LATAVG=0.0
fi
TASK=mma_${TGT}_${MODEL}_${LATVAR//./_}
. ./data_path.sh
ASR_CHECK=checkpoints/ctc_s2s_asr_${TGT}/avg_best_5_checkpoint.pt

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    --load-pretrained-encoder-from ${ASR_CHECK} \
    --config-yaml config_st.yaml \
    --train-subset distill_st \
    --valid-subset dev_st \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens 20000 \
    --update-freq 8 \
    --task speech_to_text_infer \
    --inference-config-yaml infer_st.yaml \
    --arch mma_model_s --share-decoder-input-output-embed \
    --simul-attn-type ${MODEL}_fixed_pre_decision \
    --fixed-pre-decision-ratio 8 --mass-preservation \
    --dropout 0.3 --activation-dropout 0.1 --attention-dropout 0.1 \
    --criterion mma_criterion --label-smoothing 0.1 \
    --latency-avg-weight ${LATAVG} --latency-var-weight ${LATVAR} \
    --clip-norm 10 --weight-decay 1e-6 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr 1e-3 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 --warmup-init-lr 1e-7 \
    --max-update 150000 \
    --save-dir checkpoints/${TASK} \
    --wandb-project simulst-cif-final \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --keep-last-epochs 1 \
    --keep-best-checkpoints 5 \
    --patience 20 \
    --log-format simple --log-interval 50 \
    --num-workers ${WORKERS} \
    --fp16 --fp16-init-scale 1 --memory-efficient-fp16 \
    --seed 999 \
    ${POSITIONAL[@]}
