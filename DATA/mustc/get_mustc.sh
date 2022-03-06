#!/usr/bin/env bash
SRC=en
TGT=de
DATA_ROOT=/livingrooms/george/mustc
vocab=4096
vtype=unigram

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
  python -m mustc.prep_mustc_data \
    --data-root ${DATA_ROOT} --vocab-type $vtype --vocab-size $vocab \
    --langs $TGT
fi
