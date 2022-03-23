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
5. (Optional) Preprocess data for MT (Seq-KD) with
```bash
cd DATA
bash mustc/get_data_mt.sh
```
The files should appear under `${DATA_ROOT}/en-de/mt/`.

6. Configure environment and path in `data_path.sh` before training:
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

7. (Optional) To migrate data to a new system, change paths in `scripts/migrate_data_path.sh`:
```bash
ROOT=/media/george/Data/mustc/en-de  # new data path
from=/livingrooms/george/mustc/en-de  # old data path
to=${ROOT}
```
Then run
```bash
bash scripts/migrate_data_path.sh
```