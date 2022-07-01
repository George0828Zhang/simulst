# Simultaneous Speech Translation
Code base for simultaneous speech translation experiments. It is based on [fairseq](https://github.com/pytorch/fairseq).

## Implemented
### Encoder
- [Transformers with convolutional context for ASR](https://arxiv.org/abs/1904.11660)
- [Emformer](https://arxiv.org/abs/2010.10759)

### Streaming Models
- [Wait-k](https://aclanthology.org/P19-1289) [[example](docs/waitk.md)]
- [Monotonic Multihead Attention](https://arxiv.org/abs/1909.12406) [[example](docs/mma.md)]
- [Continuous Integrate-and-Fire](https://arxiv.org/abs/1905.11235) [[example](docs/cif.md)]


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

## Pre-trained model
ASR model with Emformer encoder and Transformer decoder. Pre-trained with joint CTC cross-entropy loss.
|MuST-C (WER)|en-de (V2)|en-es|
|-|-|-|
|dev|9.65|14.44|
|tst-COMMON|12.85|14.02|
|model|[download](https://ntucc365-my.sharepoint.com/:u:/g/personal/r09922057_ntu_edu_tw/EUc3OWHv2TdDrvsj7UuUzKUBLFw0bxngdSid__81w-SYcw?e=KHg2lD)|[download](https://ntucc365-my.sharepoint.com/:u:/g/personal/r09922057_ntu_edu_tw/EVSSLkjzASVKjqEEt5NQ3oQBYhcxbT9IU1Ah0vlAuSPXww?e=grgf24)|
|vocab|[download](https://ntucc365-my.sharepoint.com/:u:/g/personal/r09922057_ntu_edu_tw/EclKBDoArG9Hv1fM5ii5KooBGUmDu13tTCJe1UYRv74rRA?e=VD7YKv)|[download](https://ntucc365-my.sharepoint.com/:u:/g/personal/r09922057_ntu_edu_tw/ESrix0mt1-BMn3UtWxxptX8BCKdCt1uldrnRhLpZd3Q1bg?e=ayq5ww)|

## Sequence-level Knowledge Distillation
|MuST-C (BLEU)|en-de (V2)|
|-|-|
|valid|31.76|
|distillation|[download](https://ntucc365-my.sharepoint.com/:u:/g/personal/r09922057_ntu_edu_tw/ER_LUQWRWatIlQkPzQh8eG0BZPOkcKoZXqPBKhxMLRuJdQ?e=iyP2NT)|
|vocab|[download](https://ntucc365-my.sharepoint.com/:u:/g/personal/r09922057_ntu_edu_tw/EclKBDoArG9Hv1fM5ii5KooBGUmDu13tTCJe1UYRv74rRA?e=VD7YKv)|


## Citation
Please consider citing our paper:
```bibtex
@article{chang2022exploring,
  title={Exploring Continuous Integrate-and-Fire for Adaptive Simultaneous Speech Translation},
  author={Chang, Chih-Chiang and Lee, Hung-yi},
  journal={arXiv e-prints},
  pages={arXiv--2204},
  year={2022}
}
```
