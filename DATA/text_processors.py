# import re
import sys
from sacremoses import MosesPunctNormalizer


class TextNormalizer:
    HTML_ESCAPES = [
        # (re.compile(r"&\s*(amp\s*;)*\s*quot\s*;"), r'"'),
        # (re.compile(r"&\s*(amp\s*;)*\s*amp\s*;"), r'"'),
        # (re.compile(r"&\s*lt\s*;"), r'<'),
        # (re.compile(r"&\s*gt\s*;"), r'>')
    ]

    def __init__(self, lang):
        self.moses_norm = MosesPunctNormalizer(
            lang=lang,
            penn=True,
            norm_quote_commas=True,
            norm_numbers=True,
            pre_replace_unicode_punct=True,
            post_remove_control_chars=True,
        )
        self.chinese = lang == 'zh'

    def __call__(self, text):
        for p, r in self.HTML_ESCAPES:
            text = p.sub(r, text)
        return self.moses_norm.normalize(text)


if __name__ == "__main__":
    p = TextNormalizer(sys.argv[1])
    for line in sys.stdin:
        print(p(line))
