#!/usr/bin/env bash
SRC=en
TGT=${1:-zh-CN}
DATA_ROOT=/livingrooms/george/covost2
vocab=5000
vtype=char
# if [ "$TGT" == "zh" ]; then
#   EXTRA="--jieba"
# fi
WORKERS=1

FAIRSEQ=~/utility/fairseq
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
source ~/envs/apex/bin/activate

OUTDIR=${DATA_ROOT}/${SRC}

# ST
feats=${OUTDIR}/fbank80.zip
if [ -f ${feats} ]; then
  echo "${feats} already exists. It is likely that you set the wrong language which is already processed."
  echo "Please change data root or clear ${feats} before continuing."
  echo "Alternatively uncomment the command below to re-process manifest only."
  # python prep_covost_data.py \
  #   --data-root ${DATA_ROOT} --vocab-type $vtype --vocab-size $vocab \
  #   --src-lang $SRC --tgt-lang $TGT --manifest-only ${EXTRA}
else
  echo "processing ${OUTDIR}"
  python prep_covost_data.py \
    --data-root ${DATA_ROOT} --vocab-type $vtype --vocab-size $vocab \
    --src-lang $SRC --tgt-lang $TGT ${EXTRA}
fi



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
