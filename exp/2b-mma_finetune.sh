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
      LAT="$2"
      shift # past argument
      shift # past value
      ;;
    -c|--ctc)
      CTC="$2"
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
LAT=${LAT:-0.0}
TASK=mma_${TGT}_${MODEL}_${LAT//./_}
. ./data_path.sh
MMA_CHECK=checkpoints/mma_${TGT}_${MODEL}_0_0/avg_best_5_checkpoint.pt

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    --finetune-from-model ${MMA_CHECK} \
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
    --latency-avg-weight ${LAT} --latency-var-weight ${LAT} \
    --clip-norm 10 --weight-decay 1e-6 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr 1e-3 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 --warmup-init-lr 1e-7 \
    --max-update 50000 \
    --save-dir checkpoints/${TASK} \
    --wandb-project simulst-cif-final \
    --best-checkpoint-metric latency \
    --keep-last-epochs 1 \
    --keep-best-checkpoints 5 \
    --patience 20 \
    --log-format simple --log-interval 50 \
    --num-workers ${WORKERS} \
    --fp16 --fp16-init-scale 1 --memory-efficient-fp16 \
    --seed 999 \
    ${POSITIONAL[@]}
