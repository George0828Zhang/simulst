TASK=mt
. ./data_path.sh
DATA_BIN=${DATA}/mt/data-bin

python -m fairseq_cli.train ${DATA_BIN} --user-dir ${USERDIR} \
	-s ${SRC} -t ${TGT} \
	--max-tokens 32000 \
	--update-freq 2 \
	--task translation_infer \
    --inference-config-yaml infer_mt.yaml \
	--arch transformer \
	--encoder-embed-dim 256 --encoder-attention-heads 4 \
	--encoder-normalize-before \
	--decoder-embed-dim 256 --decoder-attention-heads 4 \
	--decoder-normalize-before \
	--share-decoder-input-output-embed \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--clip-norm 10.0 \
	--dropout 0.3 --weight-decay 1e-4 \
	--optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --lr 5e-4 --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 --warmup-init-lr 1e-7 \
	--max-update 300000 \
	--wandb-project simulst-covost \
	--save-dir checkpoints/${TASK} \
	--no-epoch-checkpoints \
	--validate-interval 5 \
	--save-interval-updates 500 \
	--keep-interval-updates 1 \
	--keep-best-checkpoints 5 \
	--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
	--patience 75 \
	--log-format simple --log-interval 50 \
	--num-workers 4 \
	--seed 85 \
	--fp16
