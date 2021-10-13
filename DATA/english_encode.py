# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import itertools
import logging
import regex as re
import time

from data_utils import (
    load_df_from_tsv,
    save_df_to_tsv
)

logger = logging.getLogger(__name__)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--out-path", type=str, required=True)
    parser.add_argument("--lower-case", action="store_true")
    parser.add_argument("--no-punc", action="store_true")
    parser.add_argument("--reserve-word", type=str, default="")
    parser.add_argument(
        "--reserve-first-column",
        action="store_true",
        help="first column is sentence id",
    )
    ###
    parser.add_argument("--parallel-process-num", default=1, type=int)
    parser.add_argument("--logdir", default="")
    args = parser.parse_args()
    return args


def process_sent(sent, res_wrds, args):
    """
    1. reserve words
    2. lower case
    3. remove punctuation
    """

    # replace reserved words
    pattern = re.compile("(%s)" % "|".join(map(re.escape, res_wrds.keys())))
    sent = pattern.sub(lambda w: res_wrds[w.string[w.start():w.end()]], sent)

    # lowercase remove punc
    if args.no_punc:
        sent = re.sub(r'[\p{P}\p{Sm}]+', " ", sent)
    if args.lower_case:
        sent = " ".join(w if w in res_wrds.values() else w.lower()
                        for w in sent.split())
    return sent


def load_reserve_word(reserve_word):
    if reserve_word == "":
        return []
    with open(reserve_word, "r") as fp:
        res_wrds = [x.strip().split() for x in fp.readlines() if x.strip() != ""]
        assert sum([0 if len(x) == 2 else 1 for x in res_wrds]) == 0
        res_wrds = dict(res_wrds)
    return res_wrds


def process_sents(sents, args):
    out_sents = []
    res_wrds = load_reserve_word(args.reserve_word)
    for sent in sents:
        col1 = ""
        if args.reserve_first_column:
            col1, sent = sent.split(None, 1)
        sent = process_sent(sent, res_wrds, args)
        if args.reserve_first_column and col1 != "":
            sent = f"{col1} {sent}"
        out_sents.append(sent)
    return out_sents


def main():
    args = parse()
    out_sents = []
    df = load_df_from_tsv(args.data_path)
    sent_list = df["src_text"]

    if args.parallel_process_num > 1:
        try:
            import submitit
        except ImportError:
            logger.warn(
                "submitit is not found and only one job is used to process the data"
            )
            submitit = None

    if args.parallel_process_num == 1 or submitit is None:
        out_sents = process_sents(sent_list, args)
    else:
        # process sentences with parallel computation
        lsize = len(sent_list) // args.parallel_process_num + 1
        executor = submitit.AutoExecutor(folder=args.logdir)
        executor.update_parameters(timeout_min=1000, cpus_per_task=4)
        jobs = []
        for i in range(args.parallel_process_num):
            job = executor.submit(
                process_sents, sent_list[lsize * i: lsize * (i + 1)], args
            )
            jobs.append(job)
        is_running = True
        while is_running:
            time.sleep(5)
            is_running = sum([job.done() for job in jobs]) < len(jobs)
        out_sents = list(itertools.chain.from_iterable([job.result() for job in jobs]))
    # with open(args.out_path, "w") as fp:
    #     fp.write("\n".join(out_sents) + "\n")
    df["src_text"] = out_sents
    save_df_to_tsv(df, args.out_path)


if __name__ == "__main__":
    main()
