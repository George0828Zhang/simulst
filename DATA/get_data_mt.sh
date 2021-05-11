#!/usr/bin/env bash
# For MT distillation, we use the original text pair in mustc for train/dev. 
# The audio filtered set used for ST is added as testset for 
# convenience of seqKD decoding.
SRC=en
TGT=es
DATA_ROOT=/livingrooms/george/mustc
vocab=8000
vtype=unigram
workers=4

FAIRSEQ=../fairseq
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
source ~/envs/apex/bin/activate

DATA=${DATA_ROOT}/en-${TGT}
SPM_MODEL=${DATA}/spm_${vtype}${vocab}_st.model
DICT=${DATA}/spm_${vtype}${vocab}_st.txt

prep=${DATA}/mt/prep
ready=${DATA}/mt/ready
bin=${DATA}/mt/data-bin
mkdir -p $prep $ready $bin

echo "extract train, dev set..."
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

echo "extracting ST set as test..."
cut -f4 ${DATA}/train_st.tsv | tail +2 > ${prep}/test.en.raw
cut -f5 ${DATA}/train_st.tsv | tail +2 > ${prep}/test.${TGT}.raw
for lang in en ${TGT}; do
    f=${prep}/test.${lang}.raw
    tok=${prep}/test.${lang}

    if [ -f ${tok} ]; then
        echo "found ${tok}, skipping preprocess"
    else
        echo "precprocess test.${lang}"
        cat ${f} | python -m sacremoses \
        -l ${lang} -j ${workers} \
        normalize -q -d -p -c \
        tokenize -a -x  > ${tok}
    fi
done

echo "Using SPM model $SPM_MODEL"
for lang in en ${TGT}; do
    for split in train dev test; do
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
    --testpref ${ready}/test \
    --destdir ${bin} \
    --workers ${workers} \
    --joined-dictionary \
    --srcdict ${DICT}