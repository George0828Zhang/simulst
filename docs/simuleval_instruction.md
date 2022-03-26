# Evaluating using SimulEval
[SimulEval](https://github.com/facebookresearch/SimulEval) is used for evaluation.

## Install SimulEval
```
git clone https://github.com/facebookresearch/SimulEval.git
cd SimulEval
pip install -e .
```

## Extract evaluation data
The following command will generate the wav list and text file for a evaluation set `${SPLIT}` (chose from `dev`, `tst-COMMON` and `tst-HE`) in MUSTC to `${EVAL_DATA}` (default is `./data_de/`).
```
cd eval
bash 0-gen_simul_list.sh
```
The source file `${SPLIT}.wav_list` is a list of paths of audio files. Assuming your audio files stored at `/home/user/data`,
it should look like this

```bash
/home/user/data/audio-1.wav
/home/user/data/audio-2.wav
```

Each line of target file `${SPLIT}.de` is the translation for each audio file input.
```bash
Translation_1
Translation_2
```

## Long Utterances
The argument `--thresholds 20,40,60` in `0-gen_simul_list.sh` will create additional split that are called `${SPLIT}_20s`, `${SPLIT}_40s` and `${SPLIT}_60s`. These are the long utterance set that have the same content as `${SPLIT}`, but with longer, concatenated segmentation as described in paper.

## Run evaluation
```
cd eval
bash 1-simuleval.sh \
    -a ${AGENT} \
    -m ${MODEL} \
    -e ${EXP} \
    -s ${SPLIT} \
    -t ${TGT}
```

- ${AGENT}: the agent used to evaluate the model.
- ${EXP}: the experiments directory containing `checkpoints/`.
- ${MODEL}: the name of the the model (under `exp/checkpoints/`).
- \${SPLIT}: the `dev`, `tst-COMMON` or `tst-HE`. (or long utterance sets `${SPLIT}_20s`)
- ${TGT}: the target language `de`, `es`.