import math
import os
import json
import logging
import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi
import yaml
from fairseq import utils, checkpoint_utils, tasks
from fairseq.file_io import PathManager
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
try:
    from simuleval import READ_ACTION, WRITE_ACTION, DEFAULT_EOS
    from simuleval.agents import SpeechAgent
    from simuleval.states import ListEntry, SpeechStates
except ImportError:
    print("Please install simuleval 'pip install simuleval'")

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
        self.num_samples_per_shift = int(
            self.shift_size * self.sample_rate / 1000)
        self.num_samples_per_window = int(
            self.window_size * self.sample_rate / 1000)
        self.len_ms_to_samples = lambda x: x * self.sample_rate / 1000
        self.previous_residual_samples = []
        self.global_cmvn = args.global_cmvn

    def clear_cache(self):
        self.previous_residual_samples = []

    def __call__(self, new_samples):
        samples = self.previous_residual_samples + new_samples
        if len(samples) < self.num_samples_per_window:
            self.previous_residual_samples = samples
            return

        # num_frames is the number of frames from the new segment
        num_frames = math.floor(
            (len(samples) - self.len_ms_to_samples(self.window_size - self.shift_size))
            / self.num_samples_per_shift
        )

        # the number of frames used for feature extraction
        # including some part of thte previous segment
        effective_num_samples = int(
            num_frames * self.len_ms_to_samples(self.shift_size)
            + self.len_ms_to_samples(self.window_size - self.shift_size)
        )

        input_samples = samples[:effective_num_samples]
        self.previous_residual_samples = samples[
            num_frames * self.num_samples_per_shift:
        ]

        torch.manual_seed(1)
        output = kaldi.fbank(
            torch.FloatTensor(input_samples).unsqueeze(0),
            num_mel_bins=self.feature_dim,
            frame_length=self.window_size,
            frame_shift=self.shift_size,
        ).numpy()

        output = self.transform(output)

        return torch.from_numpy(output)

    def transform(self, input):
        if self.global_cmvn is None:
            return input

        mean = self.global_cmvn["mean"]
        std = self.global_cmvn["std"]

        x = np.subtract(input, mean)
        x = np.divide(x, std)
        return x


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

    speech_segment_size = 40  # in ms, 4 pooling ratio * 10 ms step size

    def __init__(self, args):
        super().__init__(args)

        self.eos = DEFAULT_EOS

        self.gpu = getattr(args, "gpu", False)

        self.args = args

        self.load_model_vocab(args)

        # if getattr(
        #     self.model.decoder.layers[0].encoder_attn,
        #     'pre_decision_ratio',
        #     None
        # ) is not None:
        #     self.speech_segment_size *= (
        #         self.model.decoder.layers[0].encoder_attn.pre_decision_ratio
        #     )
        self.speech_segment_size *= args.chunked_read
        logger.info(f"Chunked read size: {self.speech_segment_size}")

        args.global_cmvn = None
        if args.config:
            with open(os.path.join(args.data_bin, args.config), "r") as f:
                config = yaml.load(f, Loader=yaml.BaseLoader)

            if "global_cmvn" in config:
                args.global_cmvn = np.load(
                    config["global_cmvn"]["stats_npz_path"])

        if args.global_stats:
            with PathManager.open(args.global_stats, "r") as f:
                global_cmvn = json.loads(f.read())
                args.global_cmvn = {
                    "mean": global_cmvn["mean"], "std": global_cmvn["stddev"]}

        self.feature_extractor = OnlineFeatureExtractor(args)

        # self.max_len = args.max_len
        self.max_len_a = args.max_len_a
        self.max_len_b = args.max_len_b

        self.force_finish = args.force_finish

        torch.set_grad_enabled(False)

    def max_len(self, src_len):
        return self.max_len_a * src_len + self.max_len_b

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
        # parser.add_argument("--max-len", type=int, default=200,
        #                     help="Max length of translation")
        parser.add_argument("--max-len-a", type=int, default=1.2,
                            help="Max length of translation ax+b")
        parser.add_argument("--max-len-b", type=int, default=10,
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
        parser.add_argument("--chunked-read", type=int, default=1,
                            help="""chunks of 40ms (4*10ms) each READ action.
                            e.g. --chunked-read 2 means the speech encoder
                            will be updated with 80ms speech features.""")

        # fmt: on
        return parser

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
        state["cfg"]["model"].load_pretrained_encoder_from = None
        state["cfg"]["model"].load_pretrained_decoder_from = None
        self.model = task.build_model(state["cfg"]["model"])
        self.model.load_state_dict(state["model"], strict=True)
        self.model.prepare_for_inference_(state["cfg"])
        self.model.share_memory()

        if self.gpu:
            self.model.cuda()

        # Set dictionary
        self.dict = {}
        self.dict["src"] = task.source_dictionary
        self.dict["tgt"] = task.target_dictionary

    def initialize_states(self, states):
        self.feature_extractor.clear_cache()
        states.units.source = TensorListEntry()
        states.units.target = ListEntry()
        states.enc_incremental_states = {"speech": dict(), "text": dict()}
        states.dec_incremental_states = dict()
        states.speech_states = self.to_device(torch.Tensor())
        states.speech_logits = self.to_device(torch.Tensor())
        states.shrunk_states = self.to_device(torch.Tensor())
        states.finish_encode = False

    def segment_to_units(self, segment, states):
        # Convert speech samples to features
        features = self.feature_extractor(segment)
        if features is not None:
            return [features]
        else:
            return []

    def units_to_segment(self, unit_queue, states):
        """Merge sub word to full word.
        queue: stores bpe tokens.
        server: accept words.
        Therefore, we need merge subwords into word. we find the first
        subword that starts with BOW_PREFIX, then merge with subwords
        prior to this subword, remove them from queue, send to server.
        """
        # if self.segment_type == "char":
        #     return self.units_to_segment_char(unit_queue, states)
        tgt_dict = self.dict["tgt"]

        # if segment starts with eos, send EOS
        if tgt_dict.eos() == unit_queue[0]:
            return DEFAULT_EOS

        string_to_return = None

        def decode(tok_idx):
            hyp = tgt_dict.string(
                tok_idx,
                "sentencepiece",
            )
            return hyp

        # if force finish, there will be None's
        segment = []
        if None in unit_queue.value:
            unit_queue.value.remove(None)

        src_len = states.shrunk_states.size(0)
        if (
            (len(unit_queue) > 0 and tgt_dict.eos() == unit_queue[-1])
            or
            (states.finish_read() and len(states.units.target) > self.max_len(src_len))
        ):
            hyp = decode(unit_queue)
            string_to_return = ([hyp] if hyp else []) + [DEFAULT_EOS]
        else:
            space_p = None
            for p, unit_id in enumerate(unit_queue):
                if p == 0:
                    continue
                token = tgt_dict.string([unit_id])
                if token.startswith(BOW_PREFIX):
                    """
                    find the first tokens with escape symbol
                    """
                    space_p = p
                    break
            if space_p is not None:
                for j in range(space_p):
                    segment += [unit_queue.pop()]

                hyp = decode(segment)
                string_to_return = [hyp] if hyp else []

                if tgt_dict.eos() == unit_queue[0]:
                    string_to_return += [DEFAULT_EOS]

        return string_to_return

    def update_speech_encoder(self, states):
        self.print_encoder_lengths(states, prefix="BEFORE LENGTHS")
        source_len = len(states.units.source)
        speech_len = states.speech_states.size(0)
        # shrunk_len = states.shrunk_states.size(0)
        # encoder_len = 0

        # if getattr(states, "encoder_states", None) is not None:
        #     encoder_len = states.encoder_states["encoder_out"][0].size(0)

        if source_len == 0:
            return

        src_indices = self.to_device(
            states.units.source.value.unsqueeze(0)
        )
        src_lengths = self.to_device(
            torch.LongTensor([states.units.source.value.size(0)])
        )
        # states.encoder_states = self.model.encoder(src_indices, src_lengths)

        # Step 1: get new speech states
        sub_source_len = self.model.encoder.speech_encoder.subsample.get_out_seq_lens_tensor(src_lengths).item()
        encoder_out = self.model.encoder.speech_encoder(
            src_tokens=src_indices,
            src_lengths=src_lengths,
            incremental_state=states.enc_incremental_states["speech"],
            incremental_step=sub_source_len - speech_len,
        )
        encoder_out = self.model.encoder.forward_ctc_projection(encoder_out)
        logits = encoder_out["encoder_logits"][0]
        states.speech_states = torch.cat(
            (
                states.speech_states,
                encoder_out["encoder_out"][0]
            ), dim=0
        )
        states.speech_logits = torch.cat(
            (
                states.speech_logits,
                logits
            ), dim=1
        )
        torch.cuda.empty_cache()

        # print lengths
        self.print_encoder_lengths(states, prefix="ENCODER LENGTHS")

    def update_text_encoder(self, states):
        # A switch to only use this once after finish read
        if states.finish_encode:
            return

        # Step 2: discover new segs
        labels = states.speech_logits.argmax(-1).squeeze(0)
        tail = labels[-1]

        shrunk_len = states.shrunk_states.size(0)
        if states.finish_read():
            cutoff = len(labels)
        else:
            # get last label not equal to tail
            nontail = (labels != tail).nonzero()
            # no new segment?
            if nontail.numel() == 0:
                cutoff = shrunk_len
            else:
                cutoff = nontail[-1].item() + 1
        # update shrunk states
        if cutoff > shrunk_len:
            update_speech_states = states.speech_states[shrunk_len:cutoff, ...]
            update_logits = states.speech_logits[:, shrunk_len:cutoff, :]
            # Step 3: shrink new segments
            shrunk_states, shrink_lengths = self.model.encoder._weighted_shrinking_op(
                update_speech_states, update_logits)

            states.shrunk_states = torch.cat(
                (
                    states.shrunk_states,
                    shrunk_states
                ), dim=0
            )

        encoder_len = 0
        if getattr(states, "encoder_states", None) is not None:
            encoder_len = states.encoder_states["encoder_out"][0].size(0)
        # Step 4: text encoder
        shrunk_len = states.shrunk_states.size(0)
        if shrunk_len > encoder_len:
            if self.model.encoder.text_encoder is not None:
                text_out = self.model.encoder.text_encoder(
                    states.shrunk_states.transpose(0, 1),
                    incremental_state=states.enc_incremental_states["text"],
                    incremental_step=shrunk_len - encoder_len,
                )
                update_text_states = text_out["encoder_out"][0]
            else:
                update_text_states = states.shrunk_states[-(
                    shrunk_len - encoder_len):, ...]

            if getattr(states, "encoder_states", None) is None:
                states.encoder_states = {
                    # List[T x B x C]
                    "encoder_out": [update_text_states],
                }
            else:
                states.encoder_states["encoder_out"][0] = torch.cat(
                    (
                        states.encoder_states["encoder_out"][0],
                        update_text_states
                    ), dim=0
                )
            states.finish_encode = states.finish_read()
        torch.cuda.empty_cache()

    def print_encoder_lengths(self, states, prefix="ENCODER LENGTHS"):
        # print lengths
        source_len = len(states.units.source)
        speech_len = states.speech_states.size(0)
        shrunk_len = states.shrunk_states.size(0)
        encoder_len = 0

        if getattr(states, "encoder_states", None) is not None:
            encoder_len = states.encoder_states["encoder_out"][0].size(0)

        logger.debug(
            f"{prefix} SRC: {source_len}, SPH: {speech_len}, SHK: {shrunk_len}, ENC: {encoder_len}")

    def update_states_read(self, states):
        # Happens after a read action.
        # self.update_model_encoder(states)
        self.update_speech_encoder(states)
        self.update_text_encoder(states)

    def policy(self, states):
        if getattr(states, "encoder_states", None) is None:
            logger.debug(f"Action: FIRST READ (Finish={states.finish_read()})")
            # import pdb; pdb.set_trace()
            if states.finish_read():
                self.update_text_encoder(states)
            return READ_ACTION

        tgt_indices = self.to_device(
            torch.LongTensor(
                [self.model.decoder.dictionary.eos()]
                + [x for x in states.units.target.value if x is not None]
            ).unsqueeze(0)
        )

        states.dec_incremental_states["steps"] = {
            "src": states.encoder_states["encoder_out"][0].size(0),
            "tgt": 1 + len(states.units.target),
        }

        states.dec_incremental_states["online"] = {
            "only": torch.tensor(not states.finish_read())}

        x, outputs = self.model.decoder.forward(
            prev_output_tokens=tgt_indices,
            encoder_out=states.encoder_states,
            incremental_state=states.dec_incremental_states,
        )

        states.decoder_out = x

        states.decoder_out_extra = outputs

        torch.cuda.empty_cache()

        srclen = states.dec_incremental_states["steps"]["src"]
        tgtlen = states.dec_incremental_states["steps"]["tgt"] - 1
        logger.debug(
            f"srclen: {srclen}, tgtlen: {tgtlen}, Action: {['READ', 'WRITE'][outputs.action]}, {'finished' if states.finish_read() else ''}")

        if outputs.action == 0:
            if states.finish_read():
                self.update_text_encoder(states)
            return READ_ACTION
        else:
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
            self.model.decoder.clean_cache(states.dec_incremental_states)
            index = None

        # logger.debug(f"WRITE {index}")

        return index
