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
    -c|--cmvn)
      CMVN="$2"
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
AGENT=${AGENT:-"./agents/waitk_fixed_predecision_agent.py"}
EXP=${EXP:-"../exp"}
source ${EXP}/data_path.sh

SRC_FILE=${SRC_FILE:-"./data/dev.wav_list"}
TGT_FILE=${TGT_FILE:-"./data/dev.${TGT}.tok"}
CMVN=${CMVN:-"./data/gcmvn.npz"}
WAITK=${WAITK:-6000}


CHECKPOINT=${EXP}/checkpoints/${MODEL}/checkpoint_best.pt
# SPM_PREFIX=${DATA}/spm_unigram8000_st
SPM_PREFIX=${DATA}/spm_char_st_${SRC}_${TGT}

PORT=12347
WORKERS=2
BLEU_TOK=13a
UNIT=word
SEGM=word  # this is for agent to post-process the output
DATANAME=$(basename $(dirname ${DATA}))
OUTPUT=${DATANAME}_${TGT}-results/${MODEL}.test_${WAITK}
mkdir -p ${OUTPUT}

if [[ ${TGT} == "zh" ]] || [[ ${TGT} == "zh-CN" ]]; then
    BLEU_TOK=zh
    UNIT=word  # for zh/ja, we'll pre-tokenize the reference, so that latency could be evaluated as words
    SEGM=char
    NO_SPACE="--no-space"
fi

CHUNK=$(($WAITK*3))
SECONDS=0
simuleval \
    --agent ${AGENT} \
    --user-dir ${USERDIR} \
    --source ${SRC_FILE} \
    --target ${TGT_FILE} \
    --data-bin ${DATA} \
    --config config_st_${SRC}_${TGT}.yaml \
    --global-stats ${CMVN} \
    --model-path ${CHECKPOINT} \
    --tgt-splitter-path ${SPM_PREFIX}.model \
    --output ${OUTPUT} \
    --chunked-read ${CHUNK} \
    --overlap 1 \
    --incremental-encoder \
    --sacrebleu-tokenizer ${BLEU_TOK} \
    --eval-latency-unit ${UNIT} \
    --segment-type ${SEGM} \
    ${NO_SPACE} \
    --scores \
    --test-waitk ${WAITK} \
    --port ${PORT} \
    --workers ${WORKERS} \
    ${POSITIONAL[@]}
    # --full-sentence
echo "Elapsed: ${SECONDS}s"