. ./data_path.sh
echo "extracting global cmvn from train data"
python ../DATA/compute_covost2_gcmvn.py \
  --data-root ${DATA_ROOT} -s ${SRC} -t ${TGT} \
  --split train --gcmvn-max-num 2000 \
  --output ${DATA}

echo "done. see ${DATA}/gcmvn.npz"