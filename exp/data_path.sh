export SRC=en
export TGT=es
export DATA=/media/george/Data/mustc/${SRC}-${TGT}

FAIRSEQ=`realpath ../fairseq`
USERDIR=`realpath ../simultaneous_translation`
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"