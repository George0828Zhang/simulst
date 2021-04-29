# Simultaneous Speech Translation
Proposed: Learning to reorder without transcription by matching encoder and decoder hidden states.

## Setup

1. Install fairseq
```bash
mkdir -p utility && cd utility
git clone https://github.com/pytorch/fairseq.git
cd fairseq
git checkout 8b861be
python setup.py build_ext --inplace
# create symbolic links to project's root to prevent registry conflicts
cd ../../ # back to root
ln -s utility/fairseq/fairseq_cli fairseq_cli
ln -s utility/fairseq/fairseq fairseq
```
2. (Optional) [Install](docs/apex_installation.md) apex for faster mixed precision (fp16) training.
3. Install dependencies
```bash
pip install -r requirements.txt
```

## Data Preparation
This section introduces the data preparation for training and evaluation. Following will be based on MuST-C.

1. [Download](https://ict.fbk.eu/must-c/) and unpack the package.
```bash
cd ${DATA_ROOT}
tar -zxvf MUSTC_v1.0_en-de.tar.gz
```
2. In `DATA/get_mustc.sh`, set `DATA_ROOT` to the path of speech data (the directory of previous step).
3. Preprocess data with
```bash
cd DATA
bash get_mustc.sh
```
The output manifest files should appear under `${DATA_ROOT}/en-de/`. 

Configure environment and path in `exp/data_path.sh` before training:
```bash
export SRC=en
export TGT=de
export DATA=/media/george/Data/mustc/${SRC}-${TGT} # should be ${DATA_ROOT}/${SRC}-${TGT}

FAIRSEQ=../
USERDIR=../simultaneous_translation
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"

# If you have venv, add this line to use it
# source ~/envs/fair/bin/activate
```

> **_NOTE:_**  subsequent commands assume the current directory is in `exp/`.
<!-- ## Sequence-Level KD
We need a machine translation model as teacher for sequence-KD. The following command will train the nmt model with transcription and translation
```bash
bash 0-mt_distill.sh
```
Average the checkpoints to get a better model
```bash
CHECKDIR=checkpoints/mt_distill_small
CHECKPOINT_FILENAME=avg_best_5_checkpoint.pt
python ../scripts/average_checkpoints.py \
  --inputs ${CHECKDIR} --num-best-checkpoints 5 \
  --output "${CHECKDIR}/${CHECKPOINT_FILENAME}"
```
To distill the training set, run 
```bash
bash 0a-decode-distill.sh # generate prediction at ./distilled/train_st.tsv
bash 0b-create-distill-tsv.sh # generate distillation data at ${DATA_ROOT}/distill_${lang}.tsv
``` -->

## ASR Pretraining
We also need an offline ASR model to initialize our ST models. Note that the encoder arch should match the downstream st model.
```bash
bash 1-offline_asr.sh # autoregressive ASR
```
A facebook pretrained ASR for `convtransformer_espnet` can be downloaded [here](https://dl.fbaipublicfiles.com/simultaneous_translation/must_c_v1_en_de_pretrained_asr)
```bash
mkdir -p checkpoints
wget -O checkpoints/must_c_v1_en_de_pretrained_asr https://dl.fbaipublicfiles.com/simultaneous_translation/must_c_v1_en_de_pretrained_asr 
```

## Vanilla wait-k
We can now train vanilla wait-k ST model as a baseline. To do this, run
<!-- > **_NOTE:_**  to train with the distillation set, set `dataset.train_subset` to `distill_${lang}` in the script. -->
```bash
bash 2-vanilla_wait_k.sh
```

To be continued...