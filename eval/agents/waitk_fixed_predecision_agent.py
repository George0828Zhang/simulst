import math
import os
import json
import pdb
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
        self.num_samples_per_shift = int(self.shift_size * self.sample_rate / 1000)
        self.num_samples_per_window = int(self.window_size * self.sample_rate / 1000)
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
        return x.astype(np.float32)


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
        logger.debug(args)
        self.incremental_encoder = args.incremental_encoder
        self.full_sentence = args.full_sentence
        self.segment_type = args.segment_type
        self.workers = args.workers
        self.speech_segment_size *= args.chunked_read
        logger.info(f"Chunked read size: {self.speech_segment_size} ms")
        self.overlap = args.overlap
        logger.info(f"Overlap size: {self.overlap} (states)")

        if self.full_sentence:
            logger.info("Full sentence override waitk to 6000.")
            self.test_waitk = 6000

        self.eos = DEFAULT_EOS

        self.gpu = getattr(args, "gpu", False)

        self.args = args

        self.load_model_vocab(args)

        # self.speech_segment_size *= 7

        args.global_cmvn = None
        if args.global_stats:
            logger.info(f'Global CMVN: {args.global_stats}')
            args.global_cmvn = np.load(args.global_stats)
        elif args.config:
            with open(os.path.join(args.data_bin, args.config), "r") as f:
                config = yaml.load(f, Loader=yaml.BaseLoader)

            if "global_cmvn" in config:
                logger.info(f'Global CMVN: {config["global_cmvn"]["stats_npz_path"]}')
                args.global_cmvn = np.load(config["global_cmvn"]["stats_npz_path"])

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
        parser.add_argument("--test-waitk", type=int, default=1)
        parser.add_argument("--chunked-read", type=int, default=1,
                            help="""chunks of 40ms (4*10ms) each READ action.
                            e.g. --chunked-read 2 means the speech encoder
                            will be updated with 80ms speech features.""")
        parser.add_argument("--overlap", type=int, default=0,
                            help="""number of speech states to discard each
                            read action. these states contain padding information.
                            """)
        parser.add_argument("--incremental-encoder", default=False, action="store_true",
                            help="Update the model incrementally without recomputation of history.")
        parser.add_argument("--full-sentence", default=False, action="store_true",
                            help="use full sentence strategy, "
                            "by updating the encoder only once after read is finished.")
        parser.add_argument("--segment-type", type=str, default="word", choices=["word", "char"],
                            help="Agent can send a word or a char to server at a time.")
        parser.add_argument("--workers", type=int, default=1)
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
        self.model.eval()
        self.model.share_memory()

        if self.gpu:
            self.model.cuda()

        # Set dictionary
        self.dict = {}
        self.dict["tgt"] = task.target_dictionary

        self.pre_tokenizer = task.pre_tokenizer

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
        states.shrunk_length = 0

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

        src_len = states.encoder_states["encoder_out"][0].size(0)
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

    def update_speech_encoder(self, states):
        source_len = len(states.units.source)
        speech_len = states.speech_states.size(0)

        # A switch to only use this once after finish read
        if source_len == 0 or states.finish_encode:
            return

        src_indices = self.to_device(
            states.units.source.value.unsqueeze(0)
        )
        src_lengths = self.to_device(
            torch.LongTensor([states.units.source.value.size(0)])
        )

        overlap = 0 if states.finish_read() else self.overlap
        # Step 1: get new speech states
        sub_source_len = self.model.encoder.speech_encoder.subsample.get_out_seq_lens_tensor(src_lengths).item()
        inc_source_len = sub_source_len - speech_len - overlap
        if inc_source_len > 0:
            encoder_out = self.model.encoder.speech_encoder(
                src_tokens=src_indices,
                src_lengths=src_lengths,
                incremental_state=states.enc_incremental_states["speech"],
                incremental_step=sub_source_len - speech_len,
            )
            # pruning incomplete encoding output and states
            encoder_out["encoder_out"][0] = encoder_out["encoder_out"][0][:inc_source_len]
            self.model.encoder.speech_encoder.clear_cache(
                states.enc_incremental_states["speech"],
                keep=sub_source_len - overlap
            )
            encoder_out = self.model.encoder.forward_ctc_projection(encoder_out)
            states.speech_states = torch.cat(
                (
                    states.speech_states,
                    encoder_out["encoder_out"][0]
                ), dim=0
            )
            states.speech_logits = torch.cat(
                (
                    states.speech_logits,
                    encoder_out["encoder_logits"][0]
                ), dim=1
            )
            torch.cuda.empty_cache()
        else:
            logger.debug("REDUNDANT SPEECH READ")

    def update_text_encoder(self, states):
        # A switch to only use this once after finish read
        if states.finish_encode:
            return

        speech_len = states.speech_states.size(0)
        shrunk_len = states.shrunk_length

        # Step 2: discover new segs
        if states.finish_read():
            cutoff = speech_len
        elif self.model.encoder.do_weighted_shrink:
            # weighted shrinking
            labels = states.speech_logits.argmax(-1).squeeze(0)
            tail = labels[-1]

            # get last label not equal to tail
            nontail = (labels != tail).nonzero()
            # no new segment?
            if nontail.numel() == 0:
                cutoff = shrunk_len
            else:
                cutoff = nontail[-1].item() + 1
        else:
            # fixed predecision
            ratio = self.model.encoder.fixed_predecision_ratio
            cutoff = (speech_len // ratio) * ratio

        # update shrunk states
        if cutoff > shrunk_len:
            update_speech_states = states.speech_states[shrunk_len:cutoff, ...]
            update_logits = states.speech_logits[:, shrunk_len:cutoff, :]
            # Step 3: shrink new segments
            shrunk_states, shrink_lengths = self.model.encoder.shrinking_op(
                update_speech_states, update_logits)

            states.shrunk_states = torch.cat(
                (
                    states.shrunk_states,
                    shrunk_states
                ), dim=0
            )

            # remember to update shrunk len
            states.shrunk_length = shrunk_len = cutoff

        # text encoder out length
        encoder_len = 0
        if getattr(states, "encoder_states", None) is not None:
            encoder_len = states.encoder_states["encoder_out"][0].size(0)
        # Step 4: text encoder
        shrunk_state_len = states.shrunk_states.size(0)
        if shrunk_state_len > encoder_len:
            if self.model.encoder.text_encoder is not None:
                text_out = self.model.encoder.text_encoder(
                    states.shrunk_states.transpose(0, 1),
                    incremental_state=states.enc_incremental_states["text"],
                    incremental_step=shrunk_state_len - encoder_len,
                )
                update_text_states = text_out["encoder_out"][0]
            else:
                update_text_states = states.shrunk_states[-(
                    shrunk_state_len - encoder_len):, ...]

            if getattr(states, "encoder_states", None) is None:
                states.encoder_states = {
                    # List[T x B x C]
                    "encoder_out": [update_text_states],
                    "encoder_padding_mask": []
                }
            else:
                states.encoder_states["encoder_out"][0] = torch.cat(
                    (
                        states.encoder_states["encoder_out"][0],
                        update_text_states
                    ), dim=0
                )
            if (
                states.finish_read()
                and shrunk_len == speech_len
                and shrunk_state_len == states.encoder_states["encoder_out"][0].size(0)
            ):
                states.finish_encode = True
                logger.debug("FINISH ENCODE")
        else:
            logger.debug("REDUNDANT TEXT READ")
        torch.cuda.empty_cache()

    def print_encoder_lengths(self, states, prefix="ENCODER LENGTHS"):
        # print lengths
        source_len = len(states.units.source)
        speech_len = states.speech_states.size(0)
        shrunk_len = states.shrunk_length
        shrunk_state_len = states.shrunk_states.size(0)
        encoder_len = 0
        tgt_len = len(states.units.target)

        if getattr(states, "encoder_states", None) is not None:
            encoder_len = states.encoder_states["encoder_out"][0].size(0)

        finish = ", RECV_ALL" if states.finish_read() else ""

        logger.debug(
            f"{prefix} SRC: {source_len}, SPH: {shrunk_len}/{speech_len}, TXT: {encoder_len}/{shrunk_state_len}, TGT: {tgt_len}{finish}")

    def update_states_read(self, states):
        # Happens after a read action.
        if not self.full_sentence or states.finish_read():
            if self.incremental_encoder:
                self.update_speech_encoder(states)
                self.update_text_encoder(states)
            else:
                self.update_model_encoder(states)
        self.print_encoder_lengths(states, prefix="ENCODER LENGTHS")

    def policy(self, states):
        """Since we are dealing with speech, we need to waitk w.r.t
        number of source decisions, which is number of encoder states
        (which is 4x subsample of speech) divided by pre_decision ratio
        pre-decision is handled by self.speech_segment_size.
        """
        if not getattr(states, "encoder_states", None):
            # This is a rare case where source speech finished before we had enough
            # duration to compute a single text state. since simuleval will not call 
            # update_states_read if there're no new frames, we'll call it here.
            if states.finish_read():
                self.update_states_read(states)
            return READ_ACTION

        waitk = self.args.test_waitk
        src_len = states.encoder_states["encoder_out"][0].size(0)
        tgt_len = len(states.units.target)

        if src_len - tgt_len < waitk and not states.finish_read():
            # logger.info(f"Read, src_len: {src_len} tgt_len: {tgt_len}")
            return READ_ACTION
        else:
            # logger.info(f"Write, src_len: {src_len} tgt_len: {tgt_len}")
            # pdb.set_trace()
            if states.finish_read():
                # encode the last few sources (+1 eos)
                self.update_states_read(states)

            tgt_indices = self.to_device(
                torch.LongTensor(
                    [self.model.decoder.dictionary.eos()]
                    + [x for x in states.units.target.value if x is not None]
                ).unsqueeze(0)
            )

            logits, extra = self.model.forward_decoder(
                prev_output_tokens=tgt_indices,
                encoder_out=states.encoder_states,
                incremental_state=states.dec_incremental_states,
            )

            states.decoder_out = logits

            states.decoder_out_extra = extra

            torch.cuda.empty_cache()

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
            # self.model.decoder.clear_cache(states.incremental_states)
            index = None

        return index
