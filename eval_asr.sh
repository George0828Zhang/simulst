#!/usr/bin/env bash
TASK=offline_asr
SPLIT=dev #tst-COMMON
. exp/data_path.sh
CONF=$DATA/config_asr.yaml
CHECKDIR=$(pwd)/exp/checkpoints/${TASK}
AVG=true

EXTRAARGS="--scoring wer --wer-tokenizer 13a --wer-lowercase --wer-remove-punct"
GENARGS="--beam 5 --max-len-a 1.2 --max-len-b 10 --lenpen 1.1 --skip-invalid-size-inputs-valid-test"

export CUDA_VISIBLE_DEVICES=0

if [[ $AVG == "true" ]]; then
  CHECKPOINT_FILENAME=avg_best_5_checkpoint.pt
  # python ./scripts/average_checkpoints.py \
  #   --inputs ${CHECKDIR} --num-best-checkpoints 5 \
  #   --output "${CHECKDIR}/${CHECKPOINT_FILENAME}"
else
  CHECKPOINT_FILENAME=checkpoint_best.pt
fi

cd exp
python -m fairseq_cli.generate ${DATA} --user-dir .. \
  --config-yaml ${CONF} --gen-subset ${SPLIT}_asr \
  --task speech_to_text \
  --path ${CHECKDIR}/${CHECKPOINT_FILENAME} --max-tokens 80000 \
  --model-overrides '{"load_pretrained_encoder_from": None}' \
  ${GENARGS} ${EXTRAARGS}