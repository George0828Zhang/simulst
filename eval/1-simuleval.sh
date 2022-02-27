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
    -k|--waitk)
      WAITK="$2"
      shift # past argument
      shift # past value
      ;;
    -e|--expdir)
      EXP="$2"
      shift # past argument
      shift # past value
      ;;
    -s|--source)
      SRC_FILE="$2"
      shift # past argument
      shift # past value
      ;;
    -t|--target)
      TGT_FILE="$2"
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
AGENT=${AGENT:-"../codebase/agents/cif_agent.py"}
EXP=${EXP:-"../exp"}
SRC_FILE=${SRC_FILE:-"./data/dev.wav_list"}
TGT_FILE=${TGT_FILE:-"./data/dev.de"}

source ${EXP}/data_path.sh

CHECKPOINT=${EXP}/checkpoints/${MODEL}/checkpoint_best.pt
SPM_PREFIX=${DATA}/spm_unigram4096_st

PORT=12345
WORKERS=2
BLEU_TOK=13a
UNIT=word
DATANAME=$(basename $(dirname ${DATA}))
OUTPUT=${DATANAME}_${TGT}-results/${MODEL}
mkdir -p ${OUTPUT}

if [[ ${TGT} == "zh" ]]; then
    BLEU_TOK=zh
    UNIT=char
    NO_SPACE="--no-space"
fi

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