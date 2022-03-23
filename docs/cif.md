# CIF-based model

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


## Training CIF-F/CIF-IL
The training script for cif-based model is in [exp/3-cif.sh](../exp/3-cif.sh).

To train a CIF-F model with token-level quantity loss, ctc weight 0.3, use
```bash
bash 3-cif.sh -t de -q align -c 0.3 -sg
```

To train a CIF-IL model with original quantity loss, ctc weight 0.3, use
```bash
bash 3-cif.sh -t de -q sum -c 0.3 -il -sg
```

If you want to finetune from a CIF-F model with latency weight 0.5, use
```bash
bash 2b-cif_finetune.sh -t de -q align -c 0.3 -l 0.5 -sg
```

## Inference & Evaluation
The evaluation instruction is in [simuleval_instruction.md](simuleval_instruction.md).
The CIF uses the [cif_agent.py](../codebase/agents/cif_agent.py).
```
{
    "Quality": {
        "BLEU": 20.586480255254532
    },
    "Latency": {
        "AL": 1389.1190056534622,
        "AL_CA": 1587.5230795829962,
        "AP": 0.7403628496314949,
        "AP_CA": 0.78886918659345,
        "DAL": 1862.884264489703,
        "DAL_CA": 2069.6400177972478
    }
}
```