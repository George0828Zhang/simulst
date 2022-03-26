import os
import logging
import numpy as np
import torch
from fairseq import utils, checkpoint_utils, tasks
from fairseq.data.audio.audio_utils import (
    _get_kaldi_fbank, _get_torchaudio_fbank
)
logger = logging.getLogger(__name__)
try:
    from kaldi.feat import fbank
    logger.info(f"using kaldi fbank: {fbank.__file__}")
except ImportError:
    logger.info("using torchaudio fbank.")

from simuleval import READ_ACTION, WRITE_ACTION, DEFAULT_EOS
from simuleval.agents import SpeechAgent
from simuleval.states import ListEntry, SpeechStates

SHIFT_SIZE = 10
WINDOW_SIZE = 25
SAMPLE_RATE = 16000
FEATURE_DIM = 80
BOW_PREFIX = "\u2581"


class OnlineFeatureExtractor:
    """
    Extract speech feature on the fly.
    """

    def __init__(self, args):
        self.shift_size = args.shift_size
        self.window_size = args.window_size
        assert self.window_size >= self.shift_size

        self.sample_rate = args.sample_rate
        self.feature_dim = args.feature_dim
        self.num_samples_per_shift = self.shift_size * self.sample_rate // 1000
        self.num_samples_per_window = self.window_size * self.sample_rate // 1000
        self.num_samples_diff = self.num_samples_per_window - self.num_samples_per_shift
        self.previous_residual_samples = []

    def clear_cache(self):
        self.previous_residual_samples = []

    def __call__(self, new_samples):
        samples = self.previous_residual_samples + new_samples
        if len(samples) < self.num_samples_per_window:
            self.previous_residual_samples = samples
            return

        # num_frames is the number of frames from the new segment
        num_frames = (len(samples) - self.num_samples_diff) // self.num_samples_per_shift

        # the number of frames used for feature extraction
        # including some part of thte previous segment
        effective_num_samples = num_frames * self.num_samples_per_shift + self.num_samples_diff

        input_samples = samples[:effective_num_samples]
        self.previous_residual_samples = samples[
            num_frames * self.num_samples_per_shift:
        ]

        # to be consistent with extract_fbank_features
        # in DATA/data_utils.py
        _waveform = np.array([input_samples], dtype=np.float32)
        output = _get_kaldi_fbank(_waveform, self.sample_rate, self.feature_dim)
        if output is None:
            output = _get_torchaudio_fbank(_waveform, self.sample_rate, self.feature_dim)

        return torch.from_numpy(output)


class TensorListEntry(ListEntry):
    """
    Data structure to store a list of tensor.
    """

    def append(self, value):

        if len(self.value) == 0:
            self.value = value
            return

        self.value = torch.cat([self.value] + [value], dim=0)

    def info(self):
        return {
            "type": str(self.new_value_type),
            "length": self.__len__(),
            "value": "" if type(self.value) is list else self.value.size(),
        }


class FairseqSimulSTAgent(SpeechAgent):
    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--model-path', type=str, required=True,
                            help='path to your pretrained model.')
        parser.add_argument("--data-bin", type=str, required=True,
                            help="Path of data binary")
        parser.add_argument("--config", type=str, default=None,
                            help="Path to config yaml file")
        parser.add_argument("--global-stats", type=str, default=None,
                            help="Path to json file containing cmvn stats")
        parser.add_argument("--tgt-splitter-type", type=str, default="SentencePiece",
                            help="Subword splitter type for target text")
        parser.add_argument("--tgt-splitter-path", type=str, default=None,
                            help="Subword splitter model path for target text")
        parser.add_argument("--user-dir", type=str, default="examples/simultaneous_translation",
                            help="User directory for simultaneous translation")
        parser.add_argument("--max-len-a", type=int, default=1,
                            help="Max length of translation ax+b")
        parser.add_argument("--max-len-b", type=int, default=0,
                            help="Max length of translation ax+b")
        parser.add_argument("--force-finish", default=False, action="store_true",
                            help="Force the model to finish the hypothsis if the source is not finished")
        parser.add_argument("--shift-size", type=int, default=SHIFT_SIZE,
                            help="Shift size of feature extraction window.")
        parser.add_argument("--window-size", type=int, default=WINDOW_SIZE,
                            help="Window size of feature extraction window.")
        parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE,
                            help="Sample rate")
        parser.add_argument("--feature-dim", type=int, default=FEATURE_DIM,
                            help="Acoustic feature dimension.")
        parser.add_argument("--commit-unit", type=str, default="word", choices=["word", "char"],
                            help="Agent can send a word or a char to server at a time.")
        parser.add_argument("--workers", type=int, default=1)
        parser.add_argument("--debug", default=False, action="store_true")
        parser.add_argument("--full-sentence", default=False, action="store_true",
                            help="use full sentence strategy, "
                            "by updating the encoder only once after read is finished.")
        # fmt: on
        return parser

    def __init__(self, args):
        super().__init__(args)
        if args.debug:
            logger.setLevel(logging.DEBUG)

        logger.debug(args)
        self.commit_unit = args.commit_unit
        self.workers = args.workers

        self.eos = DEFAULT_EOS

        self.gpu = getattr(args, "gpu", False)

        self.args = args

        self.load_model_vocab(args)
        self.pre_decision_ratio = getattr(
            self.model.decoder.layers[0].encoder_attn,
            'pre_decision_ratio',
            1
        )
        self.full_sentence = args.full_sentence
        self.stride_ms = self.model.encoder.conv_layer_stride() * SHIFT_SIZE  # ms
        self.right_context = self.model.encoder.right_context
        self.segment_length = self.model.encoder.segment_length
        _first = (self.segment_length + self.right_context) * self.stride_ms + WINDOW_SIZE - SHIFT_SIZE
        _other = self.segment_length * self.stride_ms
        logger.info(f"First chunk: {_first} ms")
        logger.info(f"Read chunk: {_other} ms")

        self.feature_extractor = OnlineFeatureExtractor(args)

        self.max_len = lambda x: args.max_len_a * x + args.max_len_b

        self.force_finish = args.force_finish

        torch.set_grad_enabled(False)
        torch.set_num_threads(self.workers)

    def build_states(self, args, client, sentence_id):
        # Initialize states here, for example add customized entry to states
        # This function will be called at beginning of every new sentence
        states = SpeechStates(args, client, sentence_id, self)
        self.initialize_states(states)
        return states

    def to_device(self, tensor):
        if self.gpu:
            return tensor.cuda()
        else:
            return tensor.cpu()

    def load_model_vocab(self, args):
        utils.import_user_module(args)

        filename = args.model_path
        if not os.path.exists(filename):
            raise IOError("Model file not found: {}".format(filename))

        state = checkpoint_utils.load_checkpoint_to_cpu(filename)

        task_args = state["cfg"]["task"]
        task_args.data = args.data_bin

        if args.config is not None:
            task_args.config_yaml = args.config

        task = tasks.setup_task(task_args)

        # build model for ensemble
        model_args = state["cfg"]["model"]
        model_args.load_pretrained_encoder_from = None
        model_args.load_pretrained_decoder_from = None
        model_args.simul_type = None
        self.model = task.build_model(model_args)
        self.model.load_state_dict(state["model"], strict=True)
        self.model.eval()
        self.model.share_memory()

        if self.gpu:
            self.model.cuda()

        # Set dictionary
        self.dict = {}
        self.dict["tgt"] = task.target_dictionary
        self.pre_tokenizer = None

    def initialize_states(self, states):
        self.feature_extractor.clear_cache()
        states.units.source = TensorListEntry()
        states.units.target = ListEntry()
        states.enc_incremental_states = dict()
        states.dec_incremental_states = dict()

    def segment_to_units(self, segment, states):
        # Convert speech samples to features
        features = self.feature_extractor(segment)
        if features is not None:
            return [features]
        else:
            return []

    def units_to_segment(self, unit_queue, states):
        """
        queue: stores bpe tokens.
        server: accept words.

        Therefore, we need merge subwords into word. we find the first
        subword that starts with BOW_PREFIX, then merge with subwords
        prior to this subword, remove them from queue, send to server.
        """

        # Merge sub word to full word.
        tgt_dict = self.dict["tgt"]

        # if segment starts with eos, send EOS
        if tgt_dict.eos() == unit_queue[0]:
            return DEFAULT_EOS

        # if force finish, there will be None's
        segment = []
        if None in unit_queue.value:
            unit_queue.value.remove(None)

        src_len = len(states.units.source)
        if (
            (len(unit_queue) > 0 and tgt_dict.eos() == unit_queue[-1])
            or len(states.units.target) > self.max_len(src_len)
        ):
            hyp = tgt_dict.string(
                unit_queue,
                "sentencepiece",
            )
            if self.pre_tokenizer is not None:
                hyp = self.pre_tokenizer.decode(hyp)
            return [hyp] + [DEFAULT_EOS]

        for index in unit_queue:
            token = tgt_dict.string([index])
            if token.startswith(BOW_PREFIX):
                if len(segment) == 0:
                    segment += [token.replace(BOW_PREFIX, "")]
                else:
                    for j in range(len(segment)):
                        unit_queue.pop()

                    string_to_return = ["".join(segment)]

                    if tgt_dict.eos() == unit_queue[0]:
                        string_to_return += [DEFAULT_EOS]

                    return string_to_return
            else:
                segment += [token.replace(BOW_PREFIX, "")]

        return None

    def update_model_encoder(self, states):
        updated_source_len = len(states.units.source)
        update_len = updated_source_len - getattr(states, "last_update_source_len", 0)
        if update_len == 0 and states.finish_read():
            return
        finish = (update_len < self.expected_frames) or states.finish_read()
        logger.debug(f"updating {update_len} expect {self.expected_frames} {'finish' if finish else ''}")
        src_tokens = self.to_device(
            states.units.source.value.unsqueeze(0)
        )
        src_lengths = self.to_device(
            torch.LongTensor([states.units.source.value.size(0)])
        )

        encoder_out = self.model.encoder.infer(
            src_tokens,
            src_lengths,
            states.enc_incremental_states,
            finish=finish
        )

        # T B C
        new_enc_out = encoder_out['encoder_out'][0]  # might be 0

        if hasattr(states, "encoder_states"):
            new_enc_out = torch.cat([
                states.encoder_states['encoder_out'][0],
                new_enc_out
            ], dim=0)

        states.encoder_states = {
            "encoder_out": [new_enc_out],
            "encoder_padding_mask": [],  # dont use mask
            "encoder_embedding": [],
            "encoder_states": [],
            "src_tokens": [],
            "src_lengths": [],
        }
        states.last_update_source_len = updated_source_len
        torch.cuda.empty_cache()

    def update_model_encoder_fs(self, states):
        if len(states.units.source) == 0:
            return
        src_indices = self.to_device(
            states.units.source.value.unsqueeze(0)
        )
        src_lengths = self.to_device(
            torch.LongTensor([states.units.source.value.size(0)])
        )

        states.encoder_states = self.model.encoder(src_indices, src_lengths)
        torch.cuda.empty_cache()

    def update_states_read(self, states):
        # Happens after a read action.
        if not self.full_sentence:
            self.update_model_encoder(states)
        if self.full_sentence and states.finish_read():
            self.update_model_encoder_fs(states)

    def policy(self, states):
        if not hasattr(states, "encoder_states"):
            # first read
            self.expected_frames = (self.segment_length + self.right_context) * self.stride_ms // SHIFT_SIZE
            self.speech_segment_size = (
                self.segment_length + self.right_context) * self.stride_ms + WINDOW_SIZE - SHIFT_SIZE
            # Below is a rare case where source speech finished before we had enough
            # duration to compute a single text state. since simuleval will not call
            # update_states_read if there're no new frames, we'll call it here.
            if states.finish_read():
                self.update_states_read(states)

            return READ_ACTION

        enc_len = states.encoder_states["encoder_out"][0].size(0) // self.pre_decision_ratio
        dec_len = len(states.units.target)
        tgt_indices = self.to_device(
            torch.LongTensor(
                [self.model.decoder.dictionary.eos()]
                + [x for x in states.units.target.value if x is not None]
            ).unsqueeze(0)
        )

        # states.dec_incremental_states["steps"] = {
        #     "src": enc_len,
        #     "tgt": 1 + len(states.units.target),
        # }

        states.dec_incremental_states["online"] = not states.finish_read()

        x, outputs = self.model.decoder.forward(
            prev_output_tokens=tgt_indices,
            encoder_out=states.encoder_states,
            incremental_state=states.dec_incremental_states,
        )

        states.decoder_out = x

        states.decoder_out_extra = outputs

        torch.cuda.empty_cache()

        if outputs["action"] == 0:
            self.expected_frames = self.segment_length * self.stride_ms // SHIFT_SIZE
            self.speech_segment_size = self.segment_length * self.stride_ms
            logger.debug(f"READ (src={len(states.units.source)} enc={enc_len} dec={dec_len})")
            return READ_ACTION
        else:
            logger.debug(f"WRITE (src={len(states.units.source)} enc={enc_len} dec={dec_len})")
            return WRITE_ACTION

    def predict(self, states):
        decoder_states = states.decoder_out

        lprobs = self.model.get_normalized_probs(
            [decoder_states[:, -1:]], log_probs=True
        )

        index = lprobs.argmax(dim=-1)

        index = index[0, 0].item()

        if (
            self.force_finish
            and index == self.model.decoder.dictionary.eos()
            and not states.finish_read()
        ):
            # If we want to force finish the translation
            # (don't stop before finish reading), return a None
            self.model.decoder.clear_cache(states.dec_incremental_states)
            index = None

        return index

    # def update_states_read(self, states):
    #     from torch.testing import assert_close
    #     # [DEBUG] Happens after a read action.
    #     self.update_model_encoder(states)
    #     if states.finish_read():
    #         incr_out = states.encoder_states
    #         src_indices = self.to_device(
    #             states.units.source.value.unsqueeze(0)
    #         )
    #         src_lengths = self.to_device(
    #             torch.LongTensor([states.units.source.value.size(0)])
    #         )
    #         full_out = self.model.encoder(src_indices, src_lengths)

    #         def testing(key='encoder_out', Tdim=0):
    #             try:
    #                 assert_close(
    #                     incr_out[key][0],
    #                     full_out[key][0],
    #                     atol=1e-3,
    #                     rtol=1e-3,
    #                 )
    #             except AssertionError:
    #                 t = incr_out[key][0].size(Tdim)
    #                 close = torch.isclose(
    #                     incr_out[key][0],
    #                     full_out[key][0],
    #                     atol=1e-3,
    #                     rtol=1e-3
    #                 ).transpose(Tdim, 0).view(t, -1).long().prod(-1)
    #                 print("===========wrong=======", key)
    #                 print(close)
    #                 import pdb
    #                 pdb.set_trace()

    #         testing(key='encoder_out')
    #         testing(key='cif_lengths')
    #         testing(key='cif_out')

    #         print("ok")
