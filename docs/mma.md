# Wait-k / Monotonic Multihead Attention with Fixed Pre-decision Module

This is a tutorial of training and evaluating a transformer *wait-k* / *MMA* simultaneous model on MUST-C English-Germen Dataset, from [SimulMT to SimulST: Adapting Simultaneous Text Translation to End-to-End Simultaneous Speech Translation](https://www.aclweb.org/anthology/2020.aacl-main.58.pdf).

[MuST-C](https://www.aclweb.org/anthology/N19-1202) is multilingual speech-to-text translation corpus with 8-language translations on English TED talks.

## Data Preparation
See [data preparation](data_preparation.md)

## ASR Pretraining
The training script for asr is in [exp/1a-pretrain_asr.sh](../exp/1a-pretrain_asr.sh).
### Pre-trained model
ASR model with Emformer encoder and Transformer decoder. Pre-trained with joint CTC cross-entropy loss.
|MuST-C (WER)|en-de (V2)|en-es|
|-|-|-|
|dev|9.65|14.44|
|tst-COMMON|12.85|14.02|
|model|[download](https://ntucc365-my.sharepoint.com/:u:/g/personal/r09922057_ntu_edu_tw/EUc3OWHv2TdDrvsj7UuUzKUBLFw0bxngdSid__81w-SYcw?e=KHg2lD)|[download](https://ntucc365-my.sharepoint.com/:u:/g/personal/r09922057_ntu_edu_tw/EVSSLkjzASVKjqEEt5NQ3oQBYhcxbT9IU1Ah0vlAuSPXww?e=grgf24)|
|vocab|[download](https://ntucc365-my.sharepoint.com/:u:/g/personal/r09922057_ntu_edu_tw/EclKBDoArG9Hv1fM5ii5KooBGUmDu13tTCJe1UYRv74rRA?e=VD7YKv)|[download](https://ntucc365-my.sharepoint.com/:u:/g/personal/r09922057_ntu_edu_tw/ESrix0mt1-BMn3UtWxxptX8BCKdCt1uldrnRhLpZd3Q1bg?e=ayq5ww)|


## Monotonic multihead attention with fixed pre-decision module
The training script for mma is in [exp/2-mma.sh](../exp/2-mma.sh).

To train a MMA-H model with latency weight 0.1, use
```bash
bash 2-mma.sh -t de -m hard_aligned -l 0.1
```

To train a MMA-IL model with latency weight 0.1, use
```bash
bash 2-mma.sh -t de -m infinite_lookback -l 0.1
```

If you want to finetune from a offline model (latency weight = 0), use
```bash
bash 2b-mma_finetune.sh -t de -m infinite_lookback -l 0.1
```

## Inference & Evaluation
The evaluation instruction is in [simuleval_instruction.md](simuleval_instruction.md).
The MMA (and wait-k) uses the [default_agent.py](../codebase/agents/default_agent.py).
```
{
    "Quality": {
        "BLEU": 22.882280993425326
    },
    "Latency": {
        "AL": 1582.635476344213,
        "AL_CA": 1824.0610745999502,
        "AP": 0.7660114625870339,
        "AP_CA": 0.8291859397671248,
        "DAL": 2127.1755059232137,
        "DAL_CA": 2391.403942353481
    }
}
```