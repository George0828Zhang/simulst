import yaml
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--src-vocab-filename", type=str, default=None)
    parser.add_argument("--cmvn-type", choices=["global", "utterance"], default=None)

    # for asr
    parser.add_argument("--rm-bpe-tokenizer", action="store_true")
    parser.add_argument("--vocab-filename", type=str, default=None)

    args = parser.parse_args()

    with open(args.path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # print(config)

    if args.src_vocab_filename is not None:
        config["src_vocab_filename"] = args.src_vocab_filename
    if args.cmvn_type is not None:
        for split in ["_train", "*"]:
            assert "_cmvn" in config["transforms"][split][0]
            config["transforms"][split][0] = f"{args.cmvn_type}_cmvn"
    if args.rm_bpe_tokenizer and "bpe_tokenizer" in config:
        del config["bpe_tokenizer"]

    # print(config)
    with open(args.path, "w") as f:
        yaml.dump(config, f)
