export SRC=en
export TGT=zh-CN
export DATA_ROOT=/livingrooms/george/covost2
export DATA=${DATA_ROOT}/${SRC}

FAIRSEQ=`realpath ~/utility/fairseq`
USERDIR=`realpath ../simultaneous_translation`
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
. ~/envs/apex/bin/activate

export NUMEXPR_MAX_THREADS=4