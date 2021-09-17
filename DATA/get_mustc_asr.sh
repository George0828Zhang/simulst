#!/usr/bin/env bash
DATA_ROOT=/livingrooms/george/mustc
LANGS="de es fr it nl pt ro ru"
VOCAB=src_dict.txt
CONF=config_asr.yaml

UPDATE=$(realpath ../scripts/update_config.py)
FAIRSEQ=~/utility/fairseq
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
source ~/envs/apex/bin/activate

DATA_DIR=${DATA_ROOT}/joint
mkdir -p ${DATA_DIR}

# TRANSCRIPTION
BPE_TRAIN=${DATA_DIR}/transcriptions
if [ -f $BPE_TRAIN ]; then
    echo "$BPE_TRAIN found, skipping concat."
else
    for f in ${DATA_ROOT}/en-*/train_pho_st.tsv; do
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

    echo "copy asr dict for all datasets"
    for l in $LANGS; do
        f=${DATA_DIR}/${VOCAB}
        dest=${DATA_ROOT}/en-${l}/${VOCAB}
        cp ${f} ${dest}
    done
fi

# symlink
echo "create symbolic links to each languages."
for split in "dev" "tst-COMMON" "tst-HE" "train"; do
    for l in $LANGS; do
        f=${DATA_ROOT}/en-${l}/${split}_pho_st.tsv
        dest=${DATA_DIR}/${split}_${l}_asr.tsv
        ln -s ${f} ${dest}
    done
done

# copy config from first language
pattern="${DATA_ROOT}/en-*/config_st.yaml"
files=( $pattern )
cp "${files[0]}" ${DATA_DIR}/${CONF}

# update configs
for l in $LANGS; do
    python ${UPDATE} \
        --path ${DATA_ROOT}/en-${l}/config_st.yaml \
        --cmvn-type utterance \
        --src-vocab-filename ${VOCAB}
done
python ${UPDATE} \
    --path ${DATA_DIR}/${CONF} \
    --rm-bpe-tokenizer \
    --vocab-filename ${VOCAB}
