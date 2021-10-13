#!/usr/bin/env bash
DATA_ROOT=/livingrooms/george/mustc
LANGS="de es fr it nl pt ro ru"
CONF=config_asr.yaml
vocab=4000  # for english source
vtype=unigram

UPDATE=$(realpath ../scripts/update_config.py)
FAIRSEQ=~/utility/fairseq
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
source ~/envs/apex/bin/activate
spm_train=$FAIRSEQ/scripts/spm_train.py

DATA_DIR=${DATA_ROOT}/joint
mkdir -p ${DATA_DIR}

# TRANSCRIPTION
BPE_TRAIN=${DATA_DIR}/transcriptions
if [ -f $BPE_TRAIN ]; then
    echo "$BPE_TRAIN found, skipping concat."
else
    for f in ${DATA_ROOT}/en-*/train_asr_st.tsv; do
        col=$(head -1 $f | tr -s '\t' '\n' | nl -nln |  grep "src_text" | cut -f1)
        tail -n+2 $f | cut -f${col} >> $BPE_TRAIN
    done 
fi

# SPM
SPM_PREFIX=spm_${vtype}${vocab}_asr
SPM_MODEL=${SPM_PREFIX}.model
DICT=${SPM_PREFIX}.txt

if [[ ! -f ${DATA_DIR}/$SPM_MODEL ]]; then
    echo "spm_train on ${BPE_TRAIN}..."
    ccvg=1.0    
    python $spm_train --input=${BPE_TRAIN} \
        --model_prefix=${DATA_DIR}/${SPM_PREFIX} \
        --vocab_size=${vocab} \
        --character_coverage=${ccvg} \
        --model_type=${vtype} \
        --normalization_rule_name=nmt_nfkc_cf
    
    cut -f1 ${DATA_DIR}/${SPM_PREFIX}.vocab | tail -n +4 | sed "s/$/ 100/g" > ${DATA_DIR}/${DICT}
    echo "done. Total: $(cat ${DATA_DIR}/${DICT} | wc -l). first few tokens:"
    head ${DATA_DIR}/${DICT}

    echo "copy asr bpe and dict for all languages"
    for l in $LANGS; do
        for file in ${SPM_MODEL} ${DICT}; do
            src=${DATA_DIR}/${file}
            dest=${DATA_ROOT}/en-${l}/${file}
            cp ${src} ${dest}
        done
    done
fi

# symlink
echo "create symbolic links to each languages."
for split in "dev" "tst-COMMON" "tst-HE" "train"; do
    for l in $LANGS; do
        f=${DATA_ROOT}/en-${l}/${split}_asr_st.tsv
        dest=${DATA_DIR}/${split}_${l}_asr.tsv
        ln -s ${f} ${dest}
    done
done

# copy config from first language
pattern="${DATA_ROOT}/en-*/config_st.yaml"
files=( $pattern )
cp "${files[0]}" ${DATA_DIR}/${CONF}

# update configs
echo "update config for each languages."
for l in $LANGS; do
    python ${UPDATE} \
        --path ${DATA_ROOT}/en-${l}/config_st.yaml \
        --cmvn-type utterance \
        --src-bpe-tokenizer ${DATA_ROOT}/en-${l}/${SPM_MODEL} \
        --src-vocab-filename ${DICT}
done
echo "update config for joint asr."
python ${UPDATE} \
    --path ${DATA_DIR}/${CONF} \
    --bpe-tokenizer ${DATA_DIR}/${SPM_MODEL} \
    --vocab-filename ${DICT}
