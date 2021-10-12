#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
from prep_mustc_data import (
    MUSTC
)
from data_utils import (
    extract_fbank_features,
    cal_gcmvn_stats
)
import numpy as np

from tqdm import tqdm

log = logging.getLogger(__name__)


def main(args):
    root = Path(args.data_root).absolute()
    lang = args.lang
    split = args.split

    cur_root = root / f"en-{lang}"
    assert cur_root.is_dir(), (
        f"{cur_root.as_posix()} does not exist. Skipped."
    )

    dataset = MUSTC(root.as_posix(), lang, split)
    output = Path(args.output).absolute()
    output.mkdir(exist_ok=True)

    gcmvn_feature_list = []
    for waveform, sample_rate, _, text, _, utt_id in tqdm(dataset):
        if len(gcmvn_feature_list) < args.gcmvn_max_num:
            features = extract_fbank_features(waveform, sample_rate)
            gcmvn_feature_list.append(features)

    stats = cal_gcmvn_stats(gcmvn_feature_list)
    with open(output / "gcmvn.npz", "wb") as f:
        np.savez(f, mean=stats["mean"], std=stats["std"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument("--lang", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--split", required=True, choices=MUSTC.SPLITS)
    # parser.add_argument("--recompute-fbank", action="store_true",
    #                     help="Whether to recompute fbank, or use the features"
    #                          "available in data-root.")
    parser.add_argument("--gcmvn-max-num", default=150000, type=int,
                        help="Maximum number of sentences to use to estimate"
                             "global mean and variance")
    args = parser.parse_args()

    main(args)
