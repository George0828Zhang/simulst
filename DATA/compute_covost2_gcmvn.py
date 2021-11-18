#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
from prep_covost_data import (
    CoVoST
)
# from data_utils import (
from examples.speech_to_text.data_utils import (
    extract_fbank_features,
    cal_gcmvn_stats
)
import numpy as np

from tqdm import tqdm

log = logging.getLogger(__name__)


def main(args):
    root = Path(args.data_root).absolute() / args.src_lang
    if not root.is_dir():
        raise NotADirectoryError(f"{root} does not exist")

    dataset = CoVoST(root.as_posix(), args.split, args.src_lang, args.tgt_lang)
    output = Path(args.output).absolute()
    output.mkdir(exist_ok=True)

    gcmvn_feature_list = []
    for waveform, sample_rate, _, text, _, utt_id in tqdm(dataset):
        if len(gcmvn_feature_list) < args.gcmvn_max_num:
            features = extract_fbank_features(waveform, sample_rate)
            gcmvn_feature_list.append(features)
        else:
            break

    stats = cal_gcmvn_stats(gcmvn_feature_list)
    with open(output / "gcmvn.npz", "wb") as f:
        np.savez(f, mean=stats["mean"], std=stats["std"])


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
    # parser.add_argument("--recompute-fbank", action="store_true",
    #                     help="Whether to recompute fbank, or use the features"
    #                          "available in data-root.")
    parser.add_argument("--gcmvn-max-num", default=150000, type=int,
                        help="Maximum number of sentences to use to estimate"
                             "global mean and variance")
    args = parser.parse_args()

    main(args)
