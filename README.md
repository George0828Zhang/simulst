# Simultaneous Speech Translation

## Setup

1. Install fairseq
```bash
git clone https://github.com/pytorch/fairseq.git
cd fairseq
git checkout 4a7835b
python setup.py build_ext --inplace
pip install .
```
2. (Optional) [Install](docs/apex_installation.md) apex for faster mixed precision (fp16) training.
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Update submodules
```bash
git submodule update --init --recursive
```

## Data Preparation
This section introduces the data preparation for training and evaluation. Following will be based on MuST-C En-De.

1. [Download](https://ict.fbk.eu/must-c/) and unpack the package.
```bash
cd ${DATA_ROOT}
tar -zxvf MUSTC_v2.0_en-de.tar.gz
```
2. Inside `DATA/get_mustc.sh`, cnfigure the correct paths:
```bash
# the path where the data is unpacked.
DATA_ROOT=/livingrooms/george/mustc
FAIRSEQ=~/utility/fairseq
# IF NEEDED, activate your python environments
source ~/envs/apex/bin/activate
```
3. Preprocess data with
```bash
cd DATA
bash mustc/get_mustc.sh
```
The fbank and manifest files should appear under `${DATA_ROOT}/en-de/`.

4. Inside `DATA/get_data_mt.sh`, cnfigure the correct paths:
```bash
# the path where the data is unpacked.
DATA_ROOT=/livingrooms/george/mustc
FAIRSEQ=~/utility/fairseq
# IF NEEDED, activate your python environments
source ~/envs/apex/bin/activate
```
5. Preprocess data for MT with
```bash
cd DATA
bash mustc/get_data_mt.sh
```
The files should appear under `${DATA_ROOT}/en-de/mt/`.

5. Configure environment and path in `exp/data_path.sh` before training:
```bash
export SRC=en
export TGT=de
export DATA_ROOT=/livingrooms/george/mustc
export DATA=${DATA_ROOT}/${SRC}-${TGT}

FAIRSEQ=~/utility/fairseq
USERDIR=`realpath ../codebase`
export PYTHONPATH="$FAIRSEQ:$PYTHONPATH"

# IF NEEDED, activate your python environments
source ~/envs/apex/bin/activate
```

6. (Optional) To migrate data to a new system, change paths in `scripts/migrate_data_path.sh`:
```bash
ROOT=/media/george/Data/mustc/en-de  # new data path
from=/livingrooms/george/mustc/en-de  # old data path
to=${ROOT}
```
Then run
```bash
bash scripts/migrate_data_path.sh
```

## ASR Pre-training
First pre-train the ASR using joint CTC ASR
```bash
cd exp
bash 1a-pretrain_asr.sh
```
Run average checkpoint and evaluation
```bash
cd eval
bash eval_asr.sh
```

### Pre-trained model
|MuST-C|en-de(v2)|en-es|
|-|-|-|
|dev|9.65|14.44|
|model|[download](https://ntucc365-my.sharepoint.com/:u:/g/personal/r09922057_ntu_edu_tw/EUc3OWHv2TdDrvsj7UuUzKUBLFw0bxngdSid__81w-SYcw?e=KHg2lD)|[download](https://ntucc365-my.sharepoint.com/:u:/g/personal/r09922057_ntu_edu_tw/EVSSLkjzASVKjqEEt5NQ3oQBYhcxbT9IU1Ah0vlAuSPXww?e=grgf24)|
|vocab|[download](https://ntucc365-my.sharepoint.com/:u:/g/personal/r09922057_ntu_edu_tw/EclKBDoArG9Hv1fM5ii5KooBGUmDu13tTCJe1UYRv74rRA?e=VD7YKv)|[download](https://ntucc365-my.sharepoint.com/:u:/g/personal/r09922057_ntu_edu_tw/ESrix0mt1-BMn3UtWxxptX8BCKdCt1uldrnRhLpZd3Q1bg?e=ayq5ww)|

<!-- 
## MT (Seq-KD)
Train MT mode 
```bash
cd exp
bash 0-mt.sh
```
Run average checkpoint and evaluation
```bash
cd eval
bash eval_mt.sh
```

### Pre-trained model
|MuST-C|en-de(v2)|en-es|
|-|-|-|
|valid|31.76|39.86|
|model|[download]()|[download]()|
|vocab|shared w/ ST|shared w/ ST| -->
