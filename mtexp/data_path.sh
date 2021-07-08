export SRC=de
export TGT=en
export DATA=/livingrooms/george/iwslt14/de-en/data-bin

FAIRSEQ=`realpath ../fairseq`
USERDIR=`realpath ../simultaneous_translation`
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"
. ~/envs/apex/bin/activate
