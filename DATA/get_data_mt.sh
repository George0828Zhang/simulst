#!/usr/bin/env bash
# For MT distillation, we use the original text pair in mustc for train/dev. 
# The audio filtered set used for ST is added as testset for 
# convenience of seqKD decoding.
SRC=en
TGT=$1
DATA_ROOT=/livingrooms/george/mustc
vocab=8000
vtype=unigram
if [ "$TGT" == "zh" ]; then
    JIEBA=true
fi
workers=4

FAIRSEQ=~/utility/fairseq
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
source ~/envs/apex/bin/activate
SPM_ENCODE=${FAIRSEQ}/scripts/spm_encode.py
SPM_TRAIN=${FAIRSEQ}/scripts/spm_train.py
NORMALIZER=./text_processors.py
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
            if [ ${lang} = "zh" ]; then
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
	if [ ${lang} = "zh" ]; then
	    cat ${f} | \
            python $NORMALIZER zh | \
            python -m jieba_fast -q -d ' ' | \
            sed -e 's/ \{2,\}/ /g' > ${tok}
	else
	    cat ${f} | python $NORMALIZER ${lang} > ${tok}
	fi
    fi
done

echo "Training SPM for English"
SRC_SPM_PREFIX=${prep}/spm_${vtype}${vocab}_src
SRC_SPM_MODEL=${SRC_SPM_PREFIX}.model
SRC_DICT=${SRC_SPM_PREFIX}.txt

if [ -f $SRC_SPM_MODEL ]; then
    echo "SPM model: $SRC_SPM_MODEL exists, skip learning"
else
    BPE_TRAIN=${prep}/train.en
    echo "spm_train on ${BPE_TRAIN}..."
    python ${SPM_TRAIN} --input=$BPE_TRAIN \
	    --model_prefix=$SRC_SPM_PREFIX \
	    --vocab_size=$vocab \
	    --character_coverage=1.0 \
	    --model_type=$vtype \
	    --normalization_rule_name=nmt_nfkc_cf
    cut -f1 ${SRC_SPM_PREFIX}.vocab | tail -n +4 | sed "s/$/ 100/g" > ${SRC_DICT}
fi

echo "Using SRC SPM model $SRC_SPM_MODEL"
for split in train dev test; do
    f=${split}.en
    if [ -f $read/$f ]; then
	echo "found $ready/$f, skipping spm_encode"
    else
	echo "spm_encode to ${f}..."
	python ${SPM_ENCODE} --model=$SRC_SPM_MODEL \
		--output_format=piece \
		< $prep/$f > $ready/$f
    fi
done
echo "Using TGT SPM model $SPM_MODEL"
for lang in ${TGT}; do
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
    --source-lang en \
    --target-lang ${TGT} \
    --trainpref ${ready}/train \
    --validpref ${ready}/dev \
    --testpref ${ready}/test \
    --destdir ${bin} \
    --workers ${workers} \
    --srcdict ${SRC_DICT} \
    --tgtdict ${DICT}
