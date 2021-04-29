#!/usr/bin/env bash
ROOT=/media/george/Data/mustc/en-de
from1=tgt_text
to1=unused
from2=src_text
to2=tgt_text
for f in `ls ${ROOT}/*.tsv`; do
	out=${f/_st/_asr}
	echo ${f} ${out}
	sed "s~${from1}~${to1}~" $f |\
	sed "s~${from2}~${to2}~" > ${out}
done