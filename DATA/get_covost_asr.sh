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

# Extract
feats=${DATA_DIR}/fbank80.zip
if [ -f ${feats} ]; then
  echo "${feats} already exists. It is likely that you set the wrong language which is already processed."
  echo "Please change data root or clear ${feats} before continuing."
  echo "Alternatively uncomment the command below to re-process manifest only."
  # python prep_common_voice_data.py \
  #   --data-root ${DATA_ROOT} \
  #   --src-lang $SRC --manifest-only
else
  echo "processing ${DATA_DIR}"
  python prep_common_voice_data.py \
    --data-root ${DATA_ROOT} \
    --src-lang $SRC
fi

exit

# extracting phonemes
mkdir -p g2p_logdir
for split in "dev" "test" "train"; do
  echo "extract phones from ${OUTDIR}/${split}_st_${SRC}_${TGT}.tsv"
  # wc -l ${OUTDIR}/${split}_st_${SRC}_${TGT}.tsv  
  python ./g2p_encode.py \
    --parallel-process-num ${WORKERS} --logdir g2p_logdir \
    --lower-case --do-filter --use-word-start --no-punc \
    --data-path ${OUTDIR}/${split}_st_${SRC}_${TGT}.tsv \
    --out-path ${OUTDIR}/${split}_st_pho_${TGT}.tsv
done

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

    # echo "copy asr dict for all datasets"
    # for l in $LANGS; do
    #     f=${DATA_DIR}/${VOCAB}
    #     dest=${DATA_ROOT}/en-${l}/${VOCAB}
    #     cp ${f} ${dest}
    # done
fi

# # symlink
# echo "create symbolic links to each languages."
# for split in "dev" "test" "train"; do
#     for l in $LANGS; do
#         f=${DATA_ROOT}/en-${l}/${split}_pho_st.tsv
#         dest=${DATA_DIR}/${split}_${l}_asr.tsv
#         ln -s ${f} ${dest}
#     done
# done

# copy config from first language
pattern="${DATA_DIR}/config_st_*.yaml"
files=( $pattern )
cp "${files[0]}" ${DATA_DIR}/${CONF}

# update configs
for l in $LANGS; do
    python ${UPDATE} \
        --path ${DATA_DIR}/config_st_${SRC}_${l}.yaml \
        --rm-src-bpe-tokenizer \
        --src-vocab-filename ${VOCAB}
done
python ${UPDATE} \
    --path ${DATA_DIR}/${CONF} \
    --rm-bpe-tokenizer \
    --vocab-filename ${VOCAB}
