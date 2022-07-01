# Wait-k with Fixed Pre-decision Module

This is a tutorial of training and evaluating a transformer *wait-k* simultaneous model on MUST-C English-Germen Dataset, from [SimulMT to SimulST: Adapting Simultaneous Text Translation to End-to-End Simultaneous Speech Translation](https://www.aclweb.org/anthology/2020.aacl-main.58.pdf).

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


## Wait-k with fixed pre-decision module
The training script for offline waitk is in [exp/4-offline_waitk.sh](../exp/4-offline_waitk.sh).

The waitk model will be trained as an offline (wait-1024) model, and tested as a wait-1 model.
```bash
bash 4-offline_waitk.sh
```

## Inference & Evaluation
The evaluation instruction is in [simuleval_instruction.md](simuleval_instruction.md).
The wait-k uses the [default_agent.py](../codebase/agents/default_agent.py).
```
{
    "Quality": {
        "BLEU": 20.258749351223564
    },
    "Latency": {
        "AL": 1782.001343711587,
        "AL_CA": 1935.7023338036943,
        "AP": 0.7822591501150944,
        "AP_CA": 0.8479015672001843,
        "DAL": 2244.2804247360823,
        "DAL_CA": 2492.808483191793
    }
}
```