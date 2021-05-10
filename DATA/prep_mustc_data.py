#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
import string
import argparse
import logging
import os
from pathlib import Path
import shutil
from itertools import groupby
from tempfile import NamedTemporaryFile
from typing import Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from data_utils import (
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    load_df_from_tsv,
    save_df_to_tsv,
    cal_gcmvn_stats,
)
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from fairseq.data.audio.audio_utils import get_waveform
from sacremoses import MosesTokenizer, MosesPunctNormalizer

log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "src_text", "tgt_text", "speaker"]


class MUSTC(Dataset):
    """
    Create a Dataset for MuST-C. Each item is a tuple of the form:
    waveform, sample_rate, source utterance, target utterance, speaker_id,
    utterance_id
    """

    SPLITS = ["train", "dev", "tst-COMMON"]  # , "tst-HE"]
    LANGUAGES = ["de", "es", "fr", "it", "nl", "pt", "ro", "ru", "zh"]

    def __init__(self, root: str, lang: str, split: str) -> None:
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
        for wav_filename, _seg_group in groupby(segments, lambda x: x["wav"]):
            wav_path = wav_root / wav_filename
            sample_rate = sf.info(wav_path.as_posix()).samplerate
            seg_group = sorted(_seg_group, key=lambda x: x["offset"])
            for i, segment in enumerate(seg_group):
                offset = int(float(segment["offset"]) * sample_rate)
                n_frames = int(float(segment["duration"]) * sample_rate)
                _id = f"{wav_path.stem}_{i}"
                self.data.append(
                    (
                        wav_path.as_posix(),
                        offset,
                        n_frames,
                        sample_rate,
                        segment["en"],
                        segment[lang],
                        segment["speaker_id"],
                        _id,
                    )
                )

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int, str, str, str, str]:
        wav_path, offset, n_frames, sr, src_utt, tgt_utt, spk_id, utt_id = self.data[n]
        waveform, _ = get_waveform(wav_path, frames=n_frames, start=offset)
        waveform = torch.from_numpy(waveform)
        return waveform, sr, src_utt, tgt_utt, spk_id, utt_id

    def __len__(self) -> int:
        return len(self.data)

def process(args):
    root = Path(args.data_root).absolute()
    for lang in args.langs:
        cur_root = root / f"en-{lang}"
        if not cur_root.is_dir():
            print(f"{cur_root.as_posix()} does not exist. Skipped.")
            continue
        # Extract features
        feature_root = cur_root / "fbank80"
        if not args.manifest_only:
            feature_root.mkdir(exist_ok=True)
            for split in MUSTC.SPLITS:
                print(f"Fetching split {split}...")
                dataset = MUSTC(root.as_posix(), lang, split)
                print("Extracting log mel filter bank features...")
                if split == 'train' and args.cmvn_type == "global":
                    print("And estimating cepstral mean and variance stats...")
                    gcmvn_feature_list = []

                for waveform, sample_rate, _, _, _, utt_id in tqdm(dataset):
                    features = extract_fbank_features(waveform, sample_rate)

                    np.save(
                        (feature_root / f"{utt_id}.npy").as_posix(),
                        features
                    )

                    if split == 'train' and args.cmvn_type == "global":
                        if len(gcmvn_feature_list) < args.gcmvn_max_num:
                            gcmvn_feature_list.append(features)

                if split == 'train' and args.cmvn_type == "global":
                    # Estimate and save cmv
                    stats = cal_gcmvn_stats(gcmvn_feature_list)
                    with open(cur_root / "gcmvn.npz", "wb") as f:
                        np.savez(f, mean=stats["mean"], std=stats["std"])

        # Pack features into ZIP
        zip_path = cur_root / "fbank80.zip"
        if not args.manifest_only:
            print("ZIPing features...")
            create_zip(feature_root, zip_path)
        print("Fetching ZIP manifest...")
        zip_manifest = get_zip_manifest(zip_path)
        # Generate TSV manifest
        print("Generating manifest...")

        train_text = []
        for split in MUSTC.SPLITS:
            is_train_split = split.startswith("train")
            manifest = {c: [] for c in MANIFEST_COLUMNS}
            dataset = MUSTC(args.data_root, lang, split)
            for wav, sr, src_utt, tgt_utt, speaker_id, utt_id in tqdm(dataset):
                manifest["id"].append(utt_id)
                manifest["audio"].append(zip_manifest[utt_id])
                duration_ms = int(wav.size(1) / sr * 1000)
                manifest["n_frames"].append(int(1 + (duration_ms - 25) / 10))

                manifest["src_text"].append(src_utt)
                manifest["tgt_text"].append(tgt_utt)
                manifest["speaker"].append(speaker_id)
            if is_train_split:
                train_text.extend(manifest["src_text"])
                train_text.extend(manifest["tgt_text"])
            df = pd.DataFrame.from_dict(manifest)
            df = filter_manifest_df(df, is_train_split=is_train_split)
            save_df_to_tsv(df, cur_root / f"{split}_{args.task}.tsv")
        # Generate vocab
        v_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
        spm_filename_prefix = f"spm_{args.vocab_type}{v_size_str}_{args.task}"
        with NamedTemporaryFile(mode="w") as f:
            for t in train_text:
                f.write(t + "\n")
            gen_vocab(
                Path(f.name),
                cur_root / spm_filename_prefix,
                args.vocab_type,
                args.vocab_size,
            )
        # Generate config YAML
        gen_config_yaml(
            cur_root,
            spm_filename_prefix + ".model",
            yaml_filename=f"config_{args.task}.yaml",
            specaugment_policy="lb",
            cmvn_type=args.cmvn_type,
            gcmvn_path=(
                cur_root / "gcmvn.npz" if args.cmvn_type == "global"
                else None
            ),
            # pre_tokenizer={
            #     "tokenizer": "moses",
            #     "source_lang": "en",
            #     "target_lang": lang,
            #     "moses_no_dash_splits": False,
            #     "moses_no_escape": True
            # }
        )
        # Clean up
        shutil.rmtree(feature_root)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=8000, type=int)
    parser.add_argument("--task", type=str, default="st", choices=["asr", "st"])
    parser.add_argument("--langs", type=str, default=",".join(MUSTC.LANGUAGES))
    parser.add_argument("--cmvn-type", default="utterance",
                        choices=["global", "utterance"],
                        help="The type of cepstral mean and variance normalization")
    parser.add_argument("--gcmvn-max-num", default=150000, type=int,
                        help="Maximum number of sentences to use to estimate"
                             "global mean and variance")
    parser.add_argument("--manifest-only", action="store_true", help="only preprocess manifest. "
                        "works only when zip is already present"
                        )
    args = parser.parse_args()

    args.langs = args.langs.split(',')

    process(args)


if __name__ == "__main__":
    main()
