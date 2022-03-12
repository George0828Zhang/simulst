#!/usr/bin/env bash
ROOT=/media/george/Data/mustc/en-de
from=/livingrooms/george/mustc/en-de
to=${ROOT}

for f in `ls ${ROOT}/*.tsv ${ROOT}/*.yaml`; do
	echo ${f}
	sed -i "s~$from~$to~g" $f
done
