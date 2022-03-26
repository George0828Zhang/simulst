#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
from itertools import groupby
from typing import Tuple

import soundfile as sf
import torch
from torch.utils.data import Dataset
from fairseq.data.audio.audio_utils import get_waveform
from prep_mustc_data import (
    MUSTC
)

from tqdm import tqdm

log = logging.getLogger(__name__)


class LongerMUSTC(Dataset):
    """
    MuST-C w/o segmentation
    """

    SPLITS = ["train", "dev", "tst-COMMON"]  # , "tst-HE"]
    LANGUAGES = ["de", "es", "fr", "it", "nl", "pt", "ro", "ru"]

    def __init__(self, root: str, lang: str, split: str, threshold: int) -> None:
        assert split in self.SPLITS and lang in self.LANGUAGES
        _root = Path(root) / f"en-{lang}" / "data" / split
        wav_root, txt_root = _root / "wav", _root / "txt"
        assert _root.is_dir() and wav_root.is_dir() and txt_root.is_dir()
        # Load audio segments
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load the MuST-C YAML files")
        with open(txt_root / f"{split}.yaml") as f:
            segments = yaml.load(f, Loader=yaml.BaseLoader)
        # Load source and target utterances
        for _lang in ["en", lang]:
            with open(txt_root / f"{split}.{_lang}") as f:
                utterances = [r.strip() for r in f]
            assert len(segments) == len(utterances)
            for i, u in enumerate(utterances):
                segments[i][_lang] = u
        # Gather info
        self.data = []
        for wav_filename, _seg_group in tqdm(groupby(segments, lambda x: x["wav"]), desc=f"group ({threshold}s)"):
            wav_path = wav_root / wav_filename
            sample_rate = sf.info(wav_path.as_posix()).samplerate
            seg_group = sorted(_seg_group, key=lambda x: x["offset"])

            ids = []
            src_text = []
            tgt_text = []
            wavforms = []
            frames = 0

            for i, segment in enumerate(seg_group):
                offset = int(float(segment["offset"]) * sample_rate)
                n_frames = int(float(segment["duration"]) * sample_rate)
                speaker_id = segment["speaker_id"]
                waveform, _ = get_waveform(wav_path, frames=n_frames, start=offset)
                waveform = torch.from_numpy(waveform)

                ids.append(i)
                src_text.append(segment["en"])
                tgt_text.append(segment[lang])
                wavforms.append(waveform)
                frames += waveform.size(1)

                if frames / sample_rate >= threshold:
                    _id = f"{wav_path.stem}_{threshold}s_{ids[0]}_{ids[-1]}"
                    self.data.append(
                        (
                            torch.cat(wavforms, dim=1),
                            sample_rate,
                            ' '.join(src_text),
                            ' '.join(tgt_text),
                            speaker_id,
                            _id,
                        )
                    )
                    ids = []
                    src_text = []
                    tgt_text = []
                    wavforms = []
                    frames = 0

    def __getitem__(
            self, n: int
    ) -> Tuple[torch.Tensor, int, str, str, str, str]:
        return self.data[n]

    def __len__(self) -> int:
        return len(self.data)


def main(args):
    root = Path(args.data_root).absolute()
    lang = args.lang
    split = args.split

    cur_root = root / f"en-{lang}"
    assert cur_root.is_dir(), (
        f"{cur_root.as_posix()} does not exist. Skipped."
    )
    output = Path(args.output).absolute()
    output.mkdir(exist_ok=True)

    thresholds = [-1] + [int(s) for s in args.thresholds.strip().split(',')]
    for thres in thresholds:
        if thres == -1:
            suf = ""
            dataset = MUSTC(root.as_posix(), lang, split)
        else:
            suf = f"_{thres}s"
            dataset = LongerMUSTC(root.as_posix(), lang, split, thres)

        with open(output / f"{split}{suf}.{lang}", "w") as f_text, \
             open(output / f"{split}{suf}.wav_list", "w") as f_wav_list:
            for waveform, sample_rate, _, text, _, utt_id in tqdm(dataset):
                sf.write(
                    output / f"{utt_id}.wav",
                    waveform.squeeze(0).numpy(),
                    samplerate=int(sample_rate)
                )
                f_text.write(text + "\n")
                f_wav_list.write(str(output / f"{utt_id}.wav") + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument("--lang", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--split", required=True, choices=MUSTC.SPLITS)
    parser.add_argument("--thresholds", default="", type=str)
    args = parser.parse_args()

    main(args)
