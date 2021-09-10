export SRC=en
export TGT=zh
export DATA_ROOT=/livingrooms/george/mustc
export DATA=${DATA_ROOT}/${SRC}-${TGT}

FAIRSEQ=`realpath ~/utility/fairseq`
USERDIR=`realpath ../simultaneous_translation`
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
. ~/envs/apex/bin/activate