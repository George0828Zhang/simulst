#!/usr/bin/env bash
SRC=en
TGT=$1
DATA_ROOT=/livingrooms/george/mustc
vocab=8000
vtype=unigram
if [ "$TGT" == "zh" ]; then
  EXTRA="--jieba"
fi
WORKERS=1

FAIRSEQ=~/utility/fairseq
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
source ~/envs/apex/bin/activate

OUTDIR=${DATA_ROOT}/${SRC}-${TGT}

# ST
feats=${OUTDIR}/fbank80.zip
if [ -f ${feats} ]; then
  echo "${feats} already exists. It is likely that you set the wrong language which is already processed."
  echo "Please change data root or clear ${feats} before continuing."
  echo "Alternatively uncomment the command below to re-process manifest only."
  # python prep_mustc_data.py \
  #   --data-root ${DATA_ROOT} --vocab-type $vtype --vocab-size $vocab \
  #   --langs $TGT --manifest-only ${EXTRA}
else
  echo "processing ${OUTDIR}"
  python prep_mustc_data.py \
    --data-root ${DATA_ROOT} --vocab-type $vtype --vocab-size $vocab \
    --langs $TGT --cmvn-type utterance ${EXTRA}
fi


mkdir -p g2p_logdir
for split in "dev" "tst-COMMON" "tst-HE" "train"; do
  echo "extract phones for ${split}"
  python ./g2p_encode.py \
    --parallel-process-num ${WORKERS} --logdir g2p_logdir \
    --lower-case --do-filter --use-word-start --no-punc \
    --reserve-word ./mustc_noise.list \
    --data-path ${OUTDIR}/${split}_st.tsv \
    --out-path ${OUTDIR}/${split}_pho_st.tsv
done
