import argparse
import re
import sys
from pathlib import Path
import pandas as pd
from examples.speech_to_text.data_utils import load_df_from_tsv, save_df_to_tsv
from sacremoses import MosesDetokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-file", help="path to trainset tsv. typically train_st_{src}_{tgt}.tsv e.g. train_st_es_en.tsv")
    parser.add_argument("--distill-file", help="path to the file containing distillation data which "
                        "is the output of fairseq-generate. should contain lines D-{id}\tscore\t(decoded sentence).")
    parser.add_argument("--out-file", help="path to the output tsv file")
    parser.add_argument("--replace-col", default='tgt_text',
                        help="which column to replace.")
    parser.add_argument("--detok-lang", default='none',
                        help="use sacremoses to detokenize tokens into sentence. use when distill file is tokenized.")
    parser.add_argument("--verbose", action="store_true",
                        help="show warning when number of lines does not match.")

    args = parser.parse_args()

    md = None
    if args.detok_lang != "none":
        if "zh" in args.detok_lang:
            args.detok_lang = "zh"
        md = MosesDetokenizer(lang=args.detok_lang)

    train_file = Path(args.train_file)
    distill_file = Path(args.distill_file)
    out_file = Path(args.out_file)

    df = load_df_from_tsv(train_file)
    max_id = len(df.index) - 1
    skipped = []

    # D-33470	-0.07363992929458618	well...
    pattern = re.compile(r"D-(?P<id>[0-9]+)\s[-\.0-9]+\s(?P<sent>.*)$")
    lines = {}
    with open(distill_file, "r") as f:
        for line in f:
            m = pattern.match(line)
            if m is not None:
                sid = int(m.group("id"))
                sent = m.group("sent")
                if sid <= max_id:
                    lines[sid] = md.detokenize(sent.split()) if md is not None else sent
                else:
                    skipped.append(sid)

    if args.verbose:
        print(f"{len(skipped)} instances were not in tsv and were skipped. fisrt few ids: {skipped[:5]}",
              file=sys.stderr)
        print("\nsorting instances and generating new tsv...", file=sys.stderr)
    df[args.replace_col] = pd.Series(lines).sort_index()
    save_df_to_tsv(df, out_file)
