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
    -q|--qua)
      QUA="$2"
      shift # past argument
      shift # past value
      ;;
    -sg|--sg-alpha)
      POSITIONAL+=("--cif-sg-alpha") # save it in an array for later
      shift # past argument
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
QUA=${QUA:-sum}
CTC=${CTC:-0.0}
LAT=${LAT:-0.0}
TASK=cif_${TGT}_${QUA}_ctc${CTC//./_}_lat${LAT//./_}
. ./data_path.sh
CIF_CHECK=checkpoints/cif_${TGT}_${QUA}_ctc${CTC//./_}_lat0_0/avg_best_5_checkpoint.pt

python -m fairseq_cli.train ${DATA} --user-dir ${USERDIR} \
    --finetune-from-model ${CIF_CHECK} \
    --config-yaml config_st.yaml \
    --train-subset distill_st \
    --valid-subset dev_st \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens 20000 \
    --update-freq 8 \
    --task speech_to_text_infer \
    --inference-config-yaml infer_st.yaml \
    --arch cif_transformer_s --share-decoder-input-output-embed \
    --dropout 0.3 --activation-dropout 0.1 --attention-dropout 0.1 \
    --criterion cif_loss --label-smoothing 0.1 \
    --quant-type ${QUA} --ctc-factor ${CTC} --latency-factor ${LAT} \
    --clip-norm 10 --weight-decay 1e-6 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr 1e-3 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 --warmup-init-lr 1e-7 \
    --max-update 50000 \
    --save-dir checkpoints/${TASK} \
    --wandb-project simulst-cif-final \
    --keep-last-epochs 1 \
    --keep-best-checkpoints 5 \
    --patience 20 \
    --log-format simple --log-interval 50 \
    --num-workers ${WORKERS} \
    --fp16 --fp16-init-scale 1 --memory-efficient-fp16 \
    --seed 999 \
    ${POSITIONAL[@]}
