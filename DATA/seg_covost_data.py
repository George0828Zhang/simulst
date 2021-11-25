#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
import soundfile as sf
from prep_covost_data import (
    CoVoST
)

from tqdm import tqdm

logger = logging.getLogger(__name__)


def main(args):
    root = Path(args.data_root).absolute() / args.src_lang
    lang = args.tgt_lang
    split = args.split

    if not root.is_dir():
        raise NotADirectoryError(f"{root} does not exist")

    dataset = CoVoST(root.as_posix(), split, args.src_lang, lang)
    output = Path(args.output).absolute()
    output.mkdir(exist_ok=True)
    (output / "wav").mkdir(exist_ok=True)
    f_text = open(output / f"{split}.{lang}", "w")
    f_wav_list = open(output / f"{split}.wav_list", "w")
    if args.mp3:
        f_mp3_list = open(output / f"{split}.mp3_list", "w")

    too_long = []

    for i, (waveform, sample_rate, _, text, _, utt_id) in enumerate(tqdm(dataset)):
        duration_ms = int(waveform.size(1) / sample_rate * 1000)
        n_frames = int(1 + (duration_ms - 25) / 10)
        if n_frames > args.max_frames:
            too_long += [i]
            continue

        wav_path = (output / "wav" / f"{utt_id}.wav").as_posix()
        sf.write(
            wav_path,
            waveform.squeeze(0).numpy(),
            samplerate=int(sample_rate)
        )
        f_text.write(text + "\n")
        f_wav_list.write(str(wav_path) + "\n")

        if args.mp3:
            mp3_path = dataset.root / "clips" / dataset.data[i]["path"]
            f_mp3_list.write(str(mp3_path) + "\n")

    logger.info(f"| long speech (>{args.max_frames} frames): {len(too_long)} filtered, first few id: {too_long[:5]}. ")
    logger.info("Done.")

    f_text.close()
    f_wav_list.close()
    if args.mp3:
        f_mp3_list.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", "-d", required=True, type=str,
        help="data root with sub-folders for each language <root>/<src_lang>"
    )
    parser.add_argument("--src-lang", "-s", required=True, type=str)
    parser.add_argument("--tgt-lang", "-t", type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--split", required=True, choices=CoVoST.SPLITS)
    parser.add_argument("--max-frames", default=60000, type=int)
    parser.add_argument("--mp3", action="store_true")
    args = parser.parse_args()

    main(args)
