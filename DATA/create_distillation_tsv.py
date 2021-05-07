import argparse 
import re
from pathlib import Path
import pandas as pd
from data_utils import load_df_from_tsv, save_df_to_tsv
import tqdm.auto as tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", help="path to trainset tsv. typically train_{task}_{src}_{tgt}.tsv e.g. train_st_es_en.tsv")
    parser.add_argument("--distill-file", help="path to the file containing distillation data which "
    "is the output of fairseq-generate. should contain lines D-{id}\tscore\t(decoded sentence).")
    parser.add_argument("--out-file", help="path to the output tsv file")
    parser.add_argument("--replace-col", default='tgt_text',
                        help="which column to replace.")

    args = parser.parse_args()

    train_file = Path(args.train_file)
    distill_file = Path(args.distill_file)
    out_file = Path(args.out_file)

    df = load_df_from_tsv(train_file)

    pattern = re.compile(r"D-(?P<id>[0-9]+)\s[-\.0-9]+\s(?P<sent>.*)$")  # D-33470	-0.07363992929458618	well...
    lines = {}
    with open(distill_file, "r") as f:
        for line in f:
            m = pattern.match(line)
            if m is not None:
                sid = int(m.group("id"))
                sent = m.group("sent")
                lines[sid] = sent

    df[args.replace_col] = pd.Series(lines).sort_index()
    save_df_to_tsv(df, out_file)
