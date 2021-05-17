#!/usr/bin/env bash
ROOT=/media/george/Data/mustc/en-es
from=/livingrooms/george/mustc/en-es
to=/media/george/Data/mustc/en-es

for f in `ls ${ROOT}/*.tsv ${ROOT}/*.yaml`; do
	echo ${f}
	sed -i "s~$from~$to~g" $f
done
