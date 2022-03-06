export SRC=en
TGT=${TGT:-de}
export DATA_ROOT=/livingrooms/george/mustc
export DATA=${DATA_ROOT}/${SRC}-${TGT}

FAIRSEQ=`realpath ~/utility/fairseq`
USERDIR=`realpath ../codebase`
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
. ~/envs/apex/bin/activate

export NUMEXPR_MAX_THREADS=0
export WORKERS=0