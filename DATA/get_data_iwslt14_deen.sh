#!/usr/bin/env bash
# Adapted from https://github.com/pytorch/fairseq/blob/simulastsharedtask/examples/translation/prepare-iwslt14.sh
DATA_ROOT=/media/george/Data/iwslt14
FAIRSEQ=$(realpath ../fairseq)
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
source ~/envs/apex/bin/activate
SCRIPTS=~/utility/mosesdecoder/scripts

SRC=de
TGT=en
lang=de-en
vocab=8000
vtype=unigram
workers=4

TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
LC=$SCRIPTS/tokenizer/lowercase.perl

spm_train=$FAIRSEQ/scripts/spm_train.py
spm_encode=$FAIRSEQ/scripts/spm_encode.py

DATA=${DATA_ROOT}/${SRC}-${TGT}
SPM_MODEL="Current"
DICT=""

URLS=(
    "https://drive.google.com/u/0/uc?id=1Mz-iiRxRqbCVbsYfKZQ6yuAxHNzYTMnI&export=download"
    "https://drive.google.com/u/0/uc?id=1GnBarJIbNgEIIDvUyKDtLmv35Qcxg6Ed&export=download"
)
FILES=(
    "2015-01.tgz"
    "2014-01.tgz"
)

orig=${DATA}/orig
prep=${DATA}/prep
tmp=$prep/tmp
ready=${DATA}/ready
bin=${DATA}/data-bin
mkdir -p $orig $prep $tmp $ready $bin

echo "downloading data"
cd $orig
for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        gdown "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
        tar zxvf $file
        tar zxvf "${file%.*}"/texts/${SRC}/${TGT}/$lang.tgz
    fi
done
cd ..

echo "pre-processing train data..."
for l in ${SRC} ${TGT}; do
    f=train.tags.$lang.$l
    echo "precprocess train $f.$l"
    cat $orig/$lang/$f | \
        grep -v '<url>' | \
        grep -v '<talkid>' | \
        grep -v '<keywords>' | \
        grep -v '</title>' | \
        grep -v '</description>' | \
        perl $NORM_PUNC $l | \
        perl $REM_NON_PRINT_CHAR | \
        perl $TOKENIZER -threads 8 -a -l $l | \
        perl $LC > $tmp/train.tags.$lang.$l
done

echo "pre-processing test data..."
for l in ${SRC} ${TGT}; do
    for o in `ls $orig/$lang/IWSLT1*TED*.$l.xml`; do
        fname=${o##*/}
        f=$tmp/${fname%.*}
        echo $o $f
        grep '<seg id' $o | \
            sed -e 's/<seg id="[0-9]*">\s*//g' | \
            sed -e 's/\s*<\/seg>\s*//g' | \
            sed -e "s/\’/\'/g" | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 8 -l $l | \
            perl $LC > $f
        echo ""
    done
done


echo "creating train, valid, test..."
for l in ${SRC} ${TGT}; do
    awk '{if (NR%23 == 0)  print $0; }' $tmp/train.tags.$lang.$l > $prep/valid.dirty.$l
    awk '{if (NR%23 != 0)  print $0; }' $tmp/train.tags.$lang.$l > $prep/train.dirty.$l

    cat $tmp/IWSLT14.TED.dev2010.$lang.$l \
        $tmp/IWSLT14.TED.tst2010.$lang.$l \
        $tmp/IWSLT14.TED.tst2011.$lang.$l \
        $tmp/IWSLT14.TED.tst2012.$lang.$l \
        $tmp/IWSLT15.TED.tst2013.$lang.$l \
        > $prep/test.$l
done

# clean too short too long
perl $CLEAN -ratio 1.5 $prep/train.dirty ${SRC} ${TGT} $prep/train 1 1000
perl $CLEAN -ratio 1.5 $prep/valid.dirty ${SRC} ${TGT} $prep/valid 1 1000
# for l in ${SRC} ${TGT}; do
#     echo 'no cleaning applied.'
#     rm -f $prep/train.$l
#     ln -s $prep/train.dirty.$l $prep/train.$l
# done


# # SPM
if [[ -z "$SPM_MODEL" ]]; then
    echo "Don't apply SPM."
else
    if [[ "$SPM_MODEL" == "Current" ]]; then
        echo "Didn't provide SPM model, learn on current dataset"

        SPM_PREFIX=$prep/spm_${vtype}${vocab}
        SPM_MODEL=$SPM_PREFIX.model
        DICT=$SPM_PREFIX.txt
        BPE_TRAIN=$prep/all.bpe-train

        if [ -f $SPM_MODEL ]; then
            echo "SPM model: $SPM_MODEL exists, skip learning"
        else
            if [ -f $BPE_TRAIN ]; then
                echo "$BPE_TRAIN found, skipping concat."
            else
                for l in ${SRC} ${TGT}; do \
                    train=$prep/train.$l
                    valid=$prep/valid.$l
                    default=1000000
                    total=$(cat $train $valid | wc -l)
                    echo "lang $l total: $total."
                    if [ "$total" -gt "$default" ]; then
                        cat $train $valid | \
                        shuf -r -n $default >> $BPE_TRAIN
                    else
                        cat $train $valid >> $BPE_TRAIN
                    fi                    
                done
            fi

            echo "spm_train on ${BPE_TRAIN}..."
            ccvg=1.0
            if [[ ${SRC} == "zh" ]] || [[ ${TGT} == "zh" ]]; then
                ccvg=0.9995
            fi
            python $spm_train --input=$BPE_TRAIN \
                --model_prefix=$SPM_PREFIX \
                --vocab_size=$vocab \
                --character_coverage=$ccvg \
                --model_type=$vtype \
                --normalization_rule_name=nmt_nfkc_cf

            cut -f1 $SPM_PREFIX.vocab | tail -n +4 | sed "s/$/ 100/g" > $DICT
            cp $SPM_MODEL $bin/$(basename $SPM_MODEL)
            cp $DICT $bin/$(basename $DICT)
            #######################################################
        fi

    else
        if [[ ! -f $SPM_MODEL ]]; then
            echo "SPM model: $SPM_MODEL not found!"
            exit
        fi
    fi

    echo "Using SPM model $SPM_MODEL"
    for l in ${SRC} ${TGT}; do
        for f in train.$l valid.$l test.$l; do
            if [ -f $ready/$f ]; then
                echo "found $ready/$f, skipping spm_encode"
            else
                echo "spm_encode to ${f}..."
                python $spm_encode --model=$SPM_MODEL \
                    --output_format=piece \
                    < $prep/$f > $ready/$f
            fi
        done
    done
fi

python -m fairseq_cli.preprocess \
    --source-lang ${SRC} \
    --target-lang ${TGT} \
    --trainpref ${ready}/train \
    --validpref ${ready}/valid \
    --testpref ${ready}/test \
    --destdir ${bin} \
    --workers ${workers} \
    --joined-dictionary \
    --srcdict ${DICT}