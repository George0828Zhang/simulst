#!/usr/bin/env bash
# credits: https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -a|--agent)
      AGENT="$2"
      shift # past argument
      shift # past value
      ;;
    -m|--model)
      MODEL="$2"
      shift # past argument
      shift # past value
      ;;
    -e|--expdir)
      EXP="$2"
      shift # past argument
      shift # past value
      ;;
    -s|--split)
      SPLIT="$2"
      shift # past argument
      shift # past value
      ;;
    -t|--target)
      TGT="$2"
      shift # past argument
      shift # past value
      ;;
    -p|--port)
      PORT="$2"
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
PORT=${PORT:-12345}
AGENT=${AGENT:-"../codebase/agents/default_agent.py"}
EXP=${EXP:-"../exp"}
SPLIT=${SPLIT:-dev}
SRC_FILE="./data_${TGT}/${SPLIT}.wav_list"
TGT_FILE="./data_${TGT}/${SPLIT}.${TGT}"

source ${EXP}/data_path.sh

CHECKPOINT=${EXP}/checkpoints/${MODEL}/avg_best_5_checkpoint.pt
if [ ! -f ${CHECKPOINT} ]; then
    echo "avg_best_5_checkpoint.pt not found, trying checkpoint_best.pt"
    CHECKPOINT=${EXP}/checkpoints/${MODEL}/checkpoint_best.pt
fi
SPM_PREFIX=${DATA}/spm_unigram4096_st

WORKERS=1
BLEU_TOK=13a
UNIT=word
DATANAME=$(basename $(dirname ${DATA}))
OUTPUT=${DATANAME}_${TGT}_${SPLIT}-results/${MODEL}
mkdir -p ${OUTPUT}

if [[ ${TGT} == "zh" ]]; then
    BLEU_TOK=zh
    UNIT=char
    NO_SPACE="--no-space"
fi

export OMP_NUM_THREADS=${WORKERS}
export OPENBLAS_NUM_THREADS=${WORKERS}
export MKL_NUM_THREADS=${WORKERS}
export NUMEXPR_NUM_THREADS=${WORKERS}
export VECLIB_MAXIMUM_THREADS=${WORKERS}

script=$(which simuleval)
python ${script} \
    --agent ${AGENT} \
    --user-dir ${USERDIR} \
    --source ${SRC_FILE} \
    --target ${TGT_FILE} \
    --data-bin ${DATA} \
    --config config_st.yaml \
    --model-path ${CHECKPOINT} \
    --tgt-splitter-path ${SPM_PREFIX}.model \
    --output ${OUTPUT} \
    --sacrebleu-tokenizer ${BLEU_TOK} \
    --eval-latency-unit ${UNIT} \
    --commit-unit ${UNIT} \
    ${NO_SPACE} \
    --scores \
    --port ${PORT} \
    --workers ${WORKERS} \
    ${POSITIONAL[@]}