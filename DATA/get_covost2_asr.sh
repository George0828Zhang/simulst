#!/usr/bin/env bash
DATA_ROOT=/livingrooms/george/covost2
SRC=en
VOCAB=src_dict.txt
CONF=config_asr.yaml

UPDATE=$(realpath ../scripts/update_config.py)
FAIRSEQ=~/utility/fairseq
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
source ~/envs/apex/bin/activate

DATA_DIR=${DATA_ROOT}/${SRC}

# TRANSCRIPTION
BPE_TRAIN=${DATA_DIR}/transcriptions
if [ -f $BPE_TRAIN ]; then
    echo "$BPE_TRAIN found, skipping concat."
else
    for f in ${DATA_DIR}/train_st_pho_*.tsv; do
        col=$(head -1 $f | tr -s '\t' '\n' | nl -nln |  grep "src_text" | cut -f1)
        tail -n+2 $f | cut -f${col} >> $BPE_TRAIN
    done 
fi

# DICT
if [ -f ${DATA_DIR}/${VOCAB} ]; then
    echo "${DATA_DIR}/${VOCAB} found, skipping concat."
else
    wc -l $BPE_TRAIN
    echo 'estimating vocab...'
    cat $BPE_TRAIN \
        | sed 's/\s\+/\n/g' \
        | sort \
        | uniq -c \
        | sort -n -r \
        | awk '{ print $2 " " $1 }' > ${DATA_DIR}/${VOCAB}
    echo "done. Total: $(cat ${DATA_DIR}/${VOCAB} | wc -l).first few tokens:"
    head ${DATA_DIR}/${VOCAB}
fi

# copy config from first language
pattern="${DATA_DIR}/config_st_${SRC}*.yaml"
files=( $pattern )
cp "${files[0]}" ${DATA_DIR}/${CONF}

# update configs
for f in ${files[@]}; do
    python ${UPDATE} \
        --path ${f} \
        --rm-src-bpe-tokenizer \
        --src-vocab-filename ${VOCAB}
done
python ${UPDATE} \
    --path ${DATA_DIR}/${CONF} \
    --rm-bpe-tokenizer \
    --vocab-filename ${VOCAB}
