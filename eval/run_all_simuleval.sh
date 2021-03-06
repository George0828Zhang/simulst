export TGT=de
SPLIT=(
    dev
    tst-COMMON
    tst-COMMON_20s
    tst-COMMON_40s
    tst-COMMON_60s
)
CIFS=(
    cif_${TGT}_align_ctc0_3_lat0_0
    cif_${TGT}_align_ctc0_3_lat0_5
    cif_${TGT}_align_ctc0_3_lat1_0
    cif_${TGT}_align_ctc0_3_lat1_5
    cif_${TGT}_align_ctc0_3_lat2_0
    cif_${TGT}_il_sum_ctc0_3_lat0_0
    cif_${TGT}_il_sum_ctc0_3_lat0_5
    cif_${TGT}_il_sum_ctc0_3_lat1_0
    cif_${TGT}_il_sum_ctc0_3_lat1_5
    cif_${TGT}_il_sum_ctc0_3_lat2_0
)
BETAS=(
    1.0
    0.965
    0.939
    0.926
    0.933
    0.962
    0.954
    0.952
    0.949
    0.960
)
MMAS=(
    mma_${TGT}_hard_aligned_0_02
    mma_${TGT}_hard_aligned_0_04
    mma_${TGT}_hard_aligned_0_06
    mma_${TGT}_hard_aligned_0_1
    mma_${TGT}_hard_aligned_0_2
    mma_${TGT}_hard_aligned_0_4
    mma_${TGT}_infinite_lookback_0_02
    mma_${TGT}_infinite_lookback_0_04
    mma_${TGT}_infinite_lookback_0_06
    mma_${TGT}_infinite_lookback_0_1
    mma_${TGT}_infinite_lookback_0_2
    mma_${TGT}_infinite_lookback_0_4
)
function avgcheck {
    EXP=../exp
    CHECKDIR=${EXP}/checkpoints/${1}
    CHECKPOINT_FILENAME=avg_best_5_checkpoint.pt
    if [ ! -f ${CHECKDIR}/${CHECKPOINT_FILENAME} ]; then
        python ../scripts/average_checkpoints.py \
            --inputs ${CHECKDIR} --num-best-checkpoints 5 \
            --output "${CHECKDIR}/${CHECKPOINT_FILENAME}"
    fi
}
for m in "${CIFS[@]}"; do
    avgcheck ${m}
done
for m in "${MMAS[@]}"; do
    avgcheck ${m}
done

for split in "${SPLIT[@]}"; do
    for i in "${!CIFS[@]}"; do
        bash 1-simuleval.sh \
            -a ../codebase/agents/cif_agent.py \
            -m ${CIFS[i]} \
            -s ${split} \
            -t ${TGT} \
            --cif-beta ${BETAS[i]}
    done
    for m in "${MMAS[@]}"; do
        bash 1-simuleval.sh \
            -a ../codebase/agents/default_agent.py \
            -m ${m} \
            -s ${split} \
            -t ${TGT}
    done
done