# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re, spacy

spacy_vi = spacy.load('vi_core_news_lg')
SPACE_NORMALIZER = re.compile(r"\s+")


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()

def tokenize_vi(line):
    return [token.text for token in spacy_vi.tokenizer(line)]
