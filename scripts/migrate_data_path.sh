#!/usr/bin/env bash
ROOT=/media/george/Data/mustc/en-zh
from=/livingrooms/george/mustc/en-zh
to=/media/george/Data/mustc/en-zh

for f in `ls ${ROOT}/*.tsv ${ROOT}/*.yaml`; do
	echo ${f}
	sed -i "s~$from~$to~g" $f
done
