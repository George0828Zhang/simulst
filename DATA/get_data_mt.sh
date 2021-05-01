#!/usr/bin/env bash
SRC=en
TGT=de
DATA_ROOT=/media/george/Data/mustc
vocab=8000
vtype=unigram
workers=4

FAIRSEQ=../
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
# source ~/envs/apex/bin/activate

DATA=${DATA_ROOT}/en-${TGT}
SPM_MODEL=${DATA}/spm_${vtype}${vocab}_st.model
DICT=${DATA}/spm_${vtype}${vocab}_st.txt

prep=${DATA}/mt/prep
ready=${DATA}/mt/ready
bin=${DATA}/mt/data-bin
mkdir -p $prep $ready $bin

for lang in en ${TGT}; do
  for split in train dev; do
      f=${DATA}/data/${split}/txt/${split}.${lang}
      tok=${prep}/${split}.${lang}

      if [ -f ${tok} ]; then
          echo "found ${tok}, skipping preprocess"
      else
          echo "precprocess ${split}.${lang}"
          cat ${f} | python -m sacremoses \
            -l ${lang} -j ${workers} \
            normalize -q -d -p -c \
            tokenize -a -x  > ${tok}
      fi
  done
done

echo "Using SPM model $SPM_MODEL"
for lang in en ${TGT}; do
  for split in train dev; do
      f=${split}.${lang}
      if [ -f $ready/$f ]; then
          echo "found $ready/$f, skipping spm_encode"
      else
          echo "spm_encode to ${f}..."
          python spm_encode.py --model=$SPM_MODEL \
              --output_format=piece \
              < $prep/$f > $ready/$f
      fi
  done
done

python -m fairseq_cli.preprocess \
    --source-lang en \
    --target-lang ${TGT} \
    --trainpref ${ready}/train \
    --validpref ${ready}/dev \
    --destdir ${bin} \
    --workers ${workers} \
    --joined-dictionary \
    --srcdict ${DICT}