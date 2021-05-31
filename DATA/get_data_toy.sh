#!/usr/bin/env bash
# For MT distillation, we use the original text pair in mustc for train/dev. 
# The audio filtered set used for ST is added as testset for 
# convenience of seqKD decoding.
SRC=chr
TGT=num
DATA_ROOT=./toy_data
workers=4

ready=${DATA_ROOT}/ready
bin=${DATA_ROOT}/data-bin

mkdir -p ${ready}

FAIRSEQ=../fairseq
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
# source ~/envs/apex/bin/activate

echo 'generating toy data...'
python create_toy_dataset.py \
    --out-dir ${ready} \
    --n-train 300000 \
    --n-valid 3000 \
    --n-test 3000 \
    --max-len 256 \
    --min-len 20 \
    --max-upsample 4 \
    --seed 7

python -m fairseq_cli.preprocess \
    --source-lang ${SRC} \
    --target-lang ${TGT} \
    --trainpref ${ready}/train \
    --validpref ${ready}/valid \
    --testpref ${ready}/test \
    --destdir ${bin} \
    --workers ${workers} \
    --joined-dictionary