#!/usr/bin/env bash
# For MT distillation, we use the original text pair in mustc for train/dev. 
# The audio filtered set used for ST is added as testset for 
# convenience of seqKD decoding.
SRC=en
TGT=de
DATA_ROOT=/livingrooms/george/mustc
vocab=4096
vtype=unigram
workers=4

FAIRSEQ=~/utility/fairseq
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
source ~/envs/apex/bin/activate
SPM_ENCODE=${FAIRSEQ}/scripts/spm_encode.py
DATA=${DATA_ROOT}/${SRC}-${TGT}
SPM_MODEL=${DATA}/spm_${vtype}${vocab}_st.model
DICT=${DATA}/spm_${vtype}${vocab}_st.txt

prep=${DATA}/mt/prep
ready=${DATA}/mt/ready
bin=${DATA}/mt/data-bin
mkdir -p $prep $ready $bin

echo "extract train, dev set..."
for lang in ${SRC} ${TGT}; do
    for split in train dev; do
        f=${DATA}/data/${split}/txt/${split}.${lang}
        tok=${prep}/${split}.${lang}

        cp ${f} ${tok}
    done
done

echo "extracting ST set as test..."
cut -f4 ${DATA}/train_st.tsv | tail +2 > ${prep}/test.${SRC}
cut -f5 ${DATA}/train_st.tsv | tail +2 > ${prep}/test.${TGT}

echo "Using SPM model $SPM_MODEL"
for lang in ${SRC} ${TGT}; do
    for split in train dev test; do
        f=${split}.${lang}
        if [ -f $ready/$f ]; then
            echo "found $ready/$f, skipping spm_encode"
        else
            echo "spm_encode to ${f}..."
            python ${SPM_ENCODE} --model=$SPM_MODEL \
                --output_format=piece \
                < $prep/$f > $ready/$f
        fi
    done
done

python -m fairseq_cli.preprocess \
    --source-lang ${SRC} \
    --target-lang ${TGT} \
    --trainpref ${ready}/train \
    --validpref ${ready}/dev \
    --testpref ${ready}/test \
    --destdir ${bin} \
    --workers ${workers} \
    --srcdict ${DICT} \
    --joined-dictionary
