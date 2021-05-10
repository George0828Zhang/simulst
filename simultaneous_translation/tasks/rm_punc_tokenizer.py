# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import regex as re

from fairseq.data.encoders import register_tokenizer
from fairseq.dataclass import FairseqDataclass


@register_tokenizer("rm_punc", dataclass=FairseqDataclass)
class SpaceTokenizer(object):
    def __init__(self, *unused):
        self.puncs = re.compile(r'[\p{L}\p{Sm}]+')

    def encode(self, x: str) -> str:
        return self.puncs.sub(" ", x)

    def decode(self, x: str) -> str:
        return x
