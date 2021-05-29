import argparse
from pathlib import Path
import numpy as np
from tqdm import trange

def get_alphabet():
    """ credit goes to 
    https://stackoverflow.com/questions/1477294/generate-random-utf-8-string-in-python """

    try:
        get_char = unichr
    except NameError:
        get_char = chr

    include_ranges = [
        (0x0021, 0x0021),
        (0x0023, 0x0026),
        (0x0028, 0x007E),
        (0x00A1, 0x00AC),
        (0x00AE, 0x00FF),
        (0x0100, 0x017F),
        (0x0180, 0x024F),
        (0x2C60, 0x2C7F),
        (0x16A0, 0x16F0),
        (0x0370, 0x0377),
        (0x037A, 0x037E),
        (0x0384, 0x038A),
        (0x038C, 0x038C),
    ]

    alphabet = [
        get_char(code_point) for current_range in include_ranges
        for code_point in range(current_range[0], current_range[1] + 1)
    ]
    return alphabet

def generate(L, alphabet):
    src_ids = np.random.randint(0, len(alphabet), L)  # w/ random permutation
    tgt_ids = np.sort(src_ids)
    src = [alphabet[i] for i in src_ids]
    tgt = [str(i) for i in tgt_ids]
    return " ".join(src), " ".join(tgt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, help="path to output directory.")
    parser.add_argument("--n-train", type=int, default=300000, help="n training examples")
    parser.add_argument("--n-valid", type=int, default=1000, help="n validation examples")
    parser.add_argument("--n-test", type=int, default=1000, help="n testing examples")
    parser.add_argument("--max-len", type=int, default=200, help="maximum length of examples")
    parser.add_argument("--min-len", type=int, default=30, help="minimum length of examples")
    parser.add_argument("--seed", type=int, default=73, help="random seed.")

    args = parser.parse_args()

    out_dir = Path(args.out_dir).absolute()
    np.random.seed(args.seed)

    alphabet = get_alphabet()

    for split in ("train", "valid", "test"):
        srcfile = out_dir / f"{split}.chr"
        tgtfile = out_dir / f"{split}.num"
        N = getattr(args, f"n_{split}", None)
        with open(srcfile, "w") as f_src, open(tgtfile, "w") as f_tgt:
            for i in trange(N):
                L = np.random.randint(args.min_len, args.max_len + 1)
                src, tgt = generate(L, alphabet)
                f_src.write(src + "\n")
                f_tgt.write(tgt + "\n")