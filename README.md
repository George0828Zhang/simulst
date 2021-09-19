# Simultaneous Speech Translation

## Setup

1. Install fairseq
```bash
git clone https://github.com/pytorch/fairseq.git
cd fairseq
git checkout 8b861be
python setup.py build_ext --inplace
pip install .
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
tar -zxvf MUSTC_v1.2_en-zh.tar.gz
```
2. In `DATA/get_mustc.sh`, set `DATA_ROOT` to the root path of speech data (the directory of previous step).
3. Preprocess data with
```bash
cd DATA
bash get_mustc.sh ${lang}
```
The output manifest files should appear under `${DATA_ROOT}/en-${lang}/`.

4. In `DATA/get_mustc_asr.sh`, set `DATA_ROOT` and `LANGS`. Then preprocess data for joint asr with 
```bash
bash get_mustc_asr.sh
```
The output manifest files should appear under `${DATA_ROOT}/joint/`.

5. Configure environment and path in `exp/data_path.sh` before training:
```bash
export SRC=en
export TGT=de
export DATA_ROOT=/path/to/mustc/root
export DATA=${DATA_ROOT}/${SRC}-${TGT}

FAIRSEQ=/path/to/fairseq
USERDIR=`realpath ../simultaneous_translation`
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"

# If you have venv, add this line to use it
# source ~/envs/fair/bin/activate
```

## ASR Pretraining
First pretrain the speech encoder using CTC
```bash
bash 1-ctc_asr.sh
```
### Pretrained model
We provide speech encoder weights pretrained on the english transcription of MuST-C v1.0 joint data (english speech on all 8 languages). The transcriptions are converted into phoneme sequences using [g2p](DATA/g2p_encode.py). The phone error rate (PER) of the `avg_best_5_checkpoint.pt` evaluated on each split is reported below
|MuST-C Split|en-de|en-es|en-fr|en-it|en-nl|en-pt|en-ro|en-ru|Download|
|-|-|-|-|-|-|-|-|-|-|
|dev|7.894|14.250|13.890|13.486|13.940|8.318|13.882|14.181|[checkpoints](https://ntucc365-my.sharepoint.com/:u:/g/personal/r09922057_ntu_edu_tw/EXzSb9gOJXZMm7wjJCxj49gBNvMalGfTeo8zY05Cte4BUg?e=IXPjb4)|
|tst-COMMON|10.572|12.274|12.334|12.359|12.211|12.350|12.346|12.337|[src_dict.txt](https://ntucc365-my.sharepoint.com/:t:/g/personal/r09922057_ntu_edu_tw/EaZptzl7rT1Ch67JzdRXLGABUnKLy1aPbmfCnERgyITqVQ?e=28hG83)|
|tst-HE|8.573|9.339|9.317|9.507|10.332|10.153|10.046|9.532|-|
* Put `src_dict.txt` in your `${DATA_ROOT}/en-${TGT}`.

## Online Evaluation (SimulEval)
Install [SimulEval](docs/extra_installation.md).