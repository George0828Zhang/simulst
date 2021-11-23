# import re
import sys
from sacremoses import MosesPunctNormalizer
from sacrebleu.tokenizers import TOKENIZERS


class TextNormalizer:
    HTML_ESCAPES = [
        # (re.compile(r"&\s*(amp\s*;)*\s*quot\s*;"), r'"'),
        # (re.compile(r"&\s*(amp\s*;)*\s*amp\s*;"), r'"'),
        # (re.compile(r"&\s*lt\s*;"), r'<'),
        # (re.compile(r"&\s*gt\s*;"), r'>')
    ]

    def __init__(self, lang, tokenize=None):
        self.moses_norm = MosesPunctNormalizer(
            lang=lang,
            penn=True,
            norm_quote_commas=True,
            norm_numbers=True,
            pre_replace_unicode_punct=True,
            post_remove_control_chars=True,
        )
        self.tokenizer = TOKENIZERS[lang]() if tokenize else None

    def __call__(self, text):
        for p, r in self.HTML_ESCAPES:
            text = p.sub(r, text)
        text = self.moses_norm.normalize(text)
        return self.tokenizer(text) if self.tokenizer else text


if __name__ == "__main__":
    assert len(sys.argv) > 1
    p = TextNormalizer(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    for line in sys.stdin:
        print(p(line))
