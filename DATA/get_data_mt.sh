#!/usr/bin/env bash
# For MT distillation, we use the original text pair in mustc for train/dev. 
# The audio filtered set used for ST is added as testset for 
# convenience of seqKD decoding.
SRC=en
TGT=${1:-zh-CN}
DATA_ROOT=/livingrooms/george/covost2
vocab=4999
vtype=unigram
workers=4

FAIRSEQ=~/utility/fairseq
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
source ~/envs/apex/bin/activate
SPM_ENCODE=${FAIRSEQ}/scripts/spm_encode.py
SPM_TRAIN=${FAIRSEQ}/scripts/spm_train.py
NORMALIZER=./text_processors.py
DATA=${DATA_ROOT}/${SRC}
REV_DATA=${DATA_ROOT}/${TGT}

prep=${DATA}/mt/prep
ready=${DATA}/mt/ready
bin=${DATA}/mt/data-bin
mkdir -p $prep $ready $bin

echo "extract train, dev, test set..."
for split in train dev test; do
    cut -f4 ${DATA}/${split}_st_${SRC}_${TGT}.tsv | tail +2 > ${prep}/${split}.${SRC}.raw
    cut -f5 ${DATA}/${split}_st_${SRC}_${TGT}.tsv | tail +2 > ${prep}/${split}.${TGT}.raw
    if [ $split = 'train' ] && [ -f ${REV_DATA}/${split}_st_${TGT}_${SRC}.tsv ]; then
        cut -f5 ${REV_DATA}/${split}_st_${TGT}_${SRC}.tsv | tail +2 >> ${prep}/${split}.${SRC}.raw
        cut -f4 ${REV_DATA}/${split}_st_${TGT}_${SRC}.tsv | tail +2 >> ${prep}/${split}.${TGT}.raw
        echo "found and added reversed data (${TGT}->${SRC})."
    fi
    for lang in ${SRC} ${TGT}; do        
        f=${prep}/${split}.${lang}.raw
        tok=${prep}/${split}.${lang}
	
        if [ -f ${tok} ]; then
            echo "found ${tok}, skipping preprocess"
        else
            echo "precprocess ${split}.${lang}"
            if [ ${lang} = "zh-CN" ]; then
                cat ${f} | \
                python $NORMALIZER zh | \
                python -m jieba_fast -q -d ' ' | \
                    sed -e 's/ \{2,\}/ /g' > ${tok}
            else
                cat ${f} | python $NORMALIZER ${lang} > ${tok}
            fi
        fi
    done
done

for lang in ${SRC} ${TGT}; do
    echo "Training SPM for ${lang}"
    SPM_PREFIX=${prep}/spm_${vtype}${vocab}_${lang}
    SPM_MODEL=${SPM_PREFIX}.model
    DICT=${SPM_PREFIX}.txt
    if [ -f $SPM_MODEL ]; then
        echo "SPM model: $SPM_MODEL exists, skip learning"
    else
        BPE_TRAIN=${prep}/train.${lang}
        echo "spm_train on ${BPE_TRAIN}..."
        ccvg=1.0
        if [ ${lang} = "zh-CN" ]; then
            ccvg=0.9995
        fi
        python ${SPM_TRAIN} --input=$BPE_TRAIN \
            --model_prefix=$SPM_PREFIX \
            --vocab_size=$vocab \
            --character_coverage=${ccvg} \
            --model_type=$vtype \
            --normalization_rule_name=nmt_nfkc
        cut -f1 ${SPM_PREFIX}.vocab | tail -n +4 | sed "s/$/ 100/g" > ${DICT}
        cp ${SPM_MODEL} ${bin}
        cp ${DICT} ${bin}
    fi
    echo "Using SPM model $SPM_MODEL"
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
    --srcdict ${prep}/spm_${vtype}${vocab}_${SRC}.txt \
    --tgtdict ${prep}/spm_${vtype}${vocab}_${TGT}.txt
