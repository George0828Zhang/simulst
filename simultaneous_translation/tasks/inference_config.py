import logging
import os.path as op
from argparse import Namespace

logger = logging.getLogger(__name__)

class InferenceConfig(object):
    """Wrapper class for bleu config YAML"""

    def __init__(self, yaml_path):
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load YAML files for " "S2T data config")
        self.config = {}
        if op.isfile(yaml_path):
            try:
                with open(yaml_path) as f:
                    self.config = yaml.load(f, Loader=yaml.FullLoader)
            except Exception as e:
                logger.info(f"Failed to load config from {yaml_path}: {e}")
        else:
            logger.info(f"Cannot find {yaml_path}")

    @property
    def eval_wer(self):
        """evaluation with WER score in validation step."""
        return self.config.get("eval_wer", False)

    @property
    def eval_bleu(self):
        """evaluation with BLEU score in validation step."""
        return self.config.get("eval_bleu", False)    

    @property
    def eval_any(self):
        return self.eval_bleu or self.eval_wer

    @property
    def eval_bleu(self):
        """evaluation with BLEU score in validation step."""
        return self.config.get("eval_bleu", False)    

    @property
    def generation_args(self):
        """generation args for BLUE scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string"""
        args = self.config.get("generation_args", {})
        return Namespace(**args)

    @property
    def post_process(self):
        """post-process text by removing pre-processing such as BPE, letter segmentation, etc 
        (valid options are: sentencepiece, wordpiece, letter, _EOW, none, otherwise treated as BPE symbol)
        """
        return self.config.get("post_process", None)
    
    @property
    def print_samples(self):
        """print sample generations during validation"""
        return self.config.get("print_samples", False)

    @property
    def eval_bleu_args(self):
        """args for bleu scoring"""        
        args = self.config.get("eval_bleu_args", {
            "sacrebleu_tokenizer": "13a",
            "sacrebleu_lowercase": False,
            "sacrebleu_char_level": False
        })
        return Namespace(**args)

    @property
    def eval_wer_args(self):
        """args for wer scoring"""        
        args = self.config.get("eval_wer_args", {
            "wer_tokenizer": "13a",
            "wer_remove_punct": True,
            "wer_lowercase": True,
            "wer_char_level": False
        })
        return Namespace(**args)