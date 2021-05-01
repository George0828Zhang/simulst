export SRC=en
export TGT=de
export DATA=/media/george/Data/mustc/${SRC}-${TGT}

FAIRSEQ=`realpath ..`
USERDIR=`realpath ../simultaneous_translation`
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"