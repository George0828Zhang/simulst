#!/usr/bin/env bash
# Adapted from https://github.com/pytorch/fairseq/blob/simulastsharedtask/examples/translation/prepare-iwslt14.sh
DATA_ROOT=/livingrooms/george/iwslt14
FAIRSEQ=$(realpath ../fairseq)
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
source ~/envs/apex/bin/activate
SCRIPTS=~/utility/mosesdecoder/scripts
DECODED=./mt.results/generate-train.txt


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
SPM_MODEL=${DATA_ROOT}/${SRC}-${TGT}/data-bin/spm_unigram8000.model
DICT=${DATA_ROOT}/${SRC}-${TGT}/data-bin/spm_unigram8000.txt


prep=${DATA}/prep
ready=${DATA}/ready
bin=${DATA}/data-bin
newbin=${DATA}/data-bin/tmp
mkdir -p $prep $ready $bin $newbin

echo "pre-processing distill data..."
grep -E "S-[0-9]+" ${DECODED} | cut -f2 | perl $TOKENIZER -threads 8 -a -l ${SRC} > $prep/distill.${SRC}
grep -E "D-[0-9]+" ${DECODED} | cut -f3 | perl $TOKENIZER -threads 8 -a -l ${TGT} > $prep/distill.${TGT}

echo "Using SPM model $SPM_MODEL"
for l in ${SRC} ${TGT}; do
    f=distill.$l
    if [ -f $ready/$f ]; then
        echo "found $ready/$f, skipping spm_encode"
    else
        echo "spm_encode to ${f}..."
        python $spm_encode --model=$SPM_MODEL \
            --output_format=piece \
            < $prep/$f > $ready/$f
    fi
done

python -m fairseq_cli.preprocess \
    --source-lang ${SRC} \
    --target-lang ${TGT} \
    --trainpref ${ready}/distill \
    --destdir ${newbin} \
    --workers ${workers} \
    --joined-dictionary \
    --srcdict ${DICT}

for l in ${SRC} ${TGT}; do
    for ext in bin idx; do
        cp ${newbin}/train.$lang.$l.$ext ${bin}/train_distill.$lang.$l.$ext
    done
done
