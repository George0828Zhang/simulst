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
bin=${DATA}/mt/data-bin-distill
mkdir -p $prep $ready $bin

echo "extracting distill set ..."
cut -f4 ${DATA}/train_distill.tsv | tail +2 > ${prep}/train_distill.en.raw
cut -f5 ${DATA}/train_distill.tsv | tail +2 > ${prep}/train_distill.${TGT}.raw
for lang in en ${TGT}; do
    f=${prep}/train_distill.${lang}.raw
    tok=${prep}/train_distill.${lang}

    if [ -f ${tok} ]; then
        echo "found ${tok}, skipping preprocess"
    else
        echo "precprocess train_distill.${lang}"
        cat ${f} | python -m sacremoses \
        -l ${lang} -j ${workers} \
        normalize -q -d -p -c \
        tokenize -a -x  > ${tok}
    fi
done

echo "Using SPM model $SPM_MODEL"
for lang in en ${TGT}; do
    split=train_distill
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

python -m fairseq_cli.preprocess \
    --source-lang en \
    --target-lang ${TGT} \
    --trainpref ${ready}/train_distill \
    --destdir ${bin} \
    --workers ${workers} \
    --joined-dictionary \
    --srcdict ${DICT}

for lang in en ${TGT}; do
    for ext in bin idx; do
        from=${bin}/train.en-${TGT}.${lang}.${ext}
        to=${DATA}/mt/data-bin/train_distill.en-${TGT}.${lang}.${ext}
        cp ${from} ${to}
    done
done