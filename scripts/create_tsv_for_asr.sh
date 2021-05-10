#!/usr/bin/env bash
ROOT=/livingrooms/george/mustc/en-es
from1=tgt_text
to1=unused
from2=src_text
to2=tgt_text
for f in `ls ${ROOT}/*_st.tsv`; do
	out=${f/_st/_asr}
	echo ${f} ${out}
	sed "s~${from1}~${to1}~" $f |\
	sed "s~${from2}~${to2}~" > ${out}
done

echo ${ROOT}/config_st.yaml ${ROOT}/config_asr.yaml
# sed "s~target_lang: [a-z]\+~target_lang: en~" ${ROOT}/config_st.yaml > ${ROOT}/config_asr.yaml
cp ${ROOT}/config_st.yaml ${ROOT}/config_asr.yaml
echo -e "pre_tokenizer:\n  tokenizer: lc_rm" >> ${ROOT}/config_asr.yaml