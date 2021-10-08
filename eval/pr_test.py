import sys
from argparse import Namespace
from fairseq import (
    checkpoint_utils,
    tasks,
    utils,
)
from fairseq.data import data_utils
import logging
import os
import torch
import math
import pdb


def load_model(task, cfg, states, override=None, use_cuda=False, use_fp16=False):
    # mcfg = deepcopy(cfg.model)
    mcfg = vars(cfg.model)
    if override is not None:
        mcfg.update(override)
    mcfg = Namespace(**mcfg)
    model = task.build_model(mcfg)
    logger.info("loading model(s) from {}".format(checkpoint))
    model.load_state_dict(
        states["model"], strict=True, model_cfg=mcfg
    )
    if use_cuda:
        if use_fp16:
            model.half()
        model.cuda()
    model.prepare_for_inference_(cfg)
    return model


def prepare_for_scoring(generator, sample, hypothesis):
    self = generator
    src_tokens = sample["net_input"]["src_tokens"]
    bsz = src_tokens.shape[0]
    if src_tokens.dim() == 3:
        bsz, S, E = src_tokens.shape[:3]
        src_tokens = (
            src_tokens
            .unsqueeze(1)
            .expand(-1, self.beam_size, -1, -1)
            .contiguous()
            .view(bsz * self.beam_size, S, E)
        )
    else:
        src_tokens = (
            src_tokens
            .unsqueeze(1)
            .expand(-1, self.beam_size, -1)
            .contiguous()
            .view(bsz * self.beam_size, -1)
        )
    src_lengths = sample["net_input"]["src_lengths"]
    src_lengths = (
        src_lengths[:, None]
        .expand(-1, self.beam_size)
        .contiguous()
        .view(bsz * self.beam_size)
    )
    prev_output_tokens = data_utils.collate_tokens(
        [beam["tokens"] for example in hypothesis for beam in example],
        self.pad,
        self.eos,
        left_pad=False,
        move_eos_to_beginning=True,
    )
    tgt_tokens = data_utils.collate_tokens(
        [beam["tokens"] for example in hypothesis for beam in example],
        self.pad,
        self.eos,
        left_pad=False,
        move_eos_to_beginning=False,
    )
    return src_tokens, src_lengths, prev_output_tokens, tgt_tokens


if __name__ == '__main__':
    checkpoint = "../exp/checkpoints/fixed_st/checkpoint_best.pt"

    use_cuda = True
    max_tokens = 8000
    # batch_size = 2

    states = checkpoint_utils.load_checkpoint_to_cpu(
        path=checkpoint, arg_overrides=None, load_on_all_ranks=False)
    cfg = states["cfg"]
    cfg.dataset.gen_subset = "dev_pho_st"
    # cfg.dataset.batch_size = batch_size
    cfg.dataset.max_tokens = max_tokens
    cfg.model.load_pretrained_encoder_from = None
    cfg.generation.update({
        "sampling": True,
        # "sampling_topk": 100,
        "sampling_topp": 0.68,
        "beam": 5,
        # "nbest": 10,
        "max_len_a": 0.1,
        "max_len_b": 10,
    })
    cfg.common_eval.post_process = "sentencepiece"
    cfg.common_eval.results_path = "pr_test.results"

    if cfg.common_eval.results_path is not None:
        os.makedirs(cfg.common_eval.results_path, exist_ok=True)
        output_path = os.path.join(
            cfg.common_eval.results_path,
            "generate-{}.txt".format(cfg.dataset.gen_subset),
        )
        output_file = open(output_path, "w", buffering=1, encoding="utf-8")
    else:
        output_file = sys.stdout

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )
    logger = logging.getLogger("fairseq_cli.train")

    utils.import_user_module(cfg.common)

    # cuda & fp16
    use_fp16 = False
    if use_cuda:
        if torch.cuda.get_device_capability(0)[0] >= 7:
            use_fp16 = True

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)
    # Build model and criterion
    # model = task.build_model(cfg.model)
    model = load_model(
        task,
        cfg,
        states,
        use_cuda=use_cuda,
        use_fp16=use_fp16
    )
    scorer = load_model(
        task,
        cfg,
        states,
        override={
            "arch": "st2t_transformer_waitk_s",
            "waitk_list": "5",
        },
        use_cuda=use_cuda,
        use_fp16=use_fp16
    )
    criterion = task.build_criterion(cfg.criterion)
    # logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("scorer: {}".format(scorer.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info("generation: {}".format(cfg.generation))

    task.load_dataset(cfg.dataset.gen_subset, task_cfg=cfg.task)
    # Load dataset
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            # *[m.max_positions() for m in models]
            task.max_positions(), model.max_positions()
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)

    generator = task.build_generator(
        [model], cfg.generation, extra_gen_cls_kwargs=None
    )

    def apply_half(t):
        if t.dtype is torch.float32:
            return t.half()
        return t

    # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(cfg.tokenizer)

    def decode_fn(x):
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    tgt_dict = task.tgt_dict

    for _, sample in enumerate(itr):
        if use_cuda:
            sample = utils.move_to_cuda(sample)
            if use_fp16:
                sample = utils.apply_to_sample(apply_half, sample)
        if "net_input" not in sample:
            continue

        sample = task.process_sample(sample)
        hypos = task.inference_step(generator, [model], sample)
        src_tokens, src_lengths, prev_output_tokens, tgt_tokens = prepare_for_scoring(
            generator, sample, hypos)

        net_output = scorer.forward(
            src_tokens,
            src_lengths,
            prev_output_tokens,
        )
        scores = scorer.get_normalized_probs(
            net_output, log_probs=True, sample=None,
        )
        scores = scores.gather(2, tgt_tokens.unsqueeze(2))
        masks = tgt_tokens.ne(tgt_dict.pad())
        scores = (
            scores[:, :, 0].masked_fill_(~masks, 0).sum(1)
        )
        scores = scores / masks.sum(1).type_as(scores)
        # masks shape (bsz*beam, len)
        # scores shape (bsz*beam)
        top_idx = scores.view(-1, generator.beam_size).argmax(-1)

        for i, sample_id in enumerate(sample["id"].tolist()):
            target_tokens = (
                utils.strip_pad(sample["target"][i, :],
                                tgt_dict.pad()).int().cpu()
            )
            target_str = tgt_dict.string(
                target_tokens,
                cfg.common_eval.post_process,
                escape_unk=True,
                extra_symbols_to_ignore={},
            )
            target_str = decode_fn(target_str)

            print("T-{}\t{}".format(sample_id, target_str), file=output_file)

            # Process top prediction
            def process_prediction(hypo):
                hypo_tokens = (
                    hypo["tokens"].int().cpu()
                )
                hypo_str = tgt_dict.string(
                    hypo_tokens,
                    cfg.common_eval.post_process,
                    escape_unk=True,
                    extra_symbols_to_ignore={},
                )
                detok_hypo_str = decode_fn(hypo_str)
                score = hypo["score"] / math.log(2)  # convert to base 2
                # original hypothesis (after tokenization and BPE)
                print(
                    "H-{}\t{}\t{}".format(sample_id, score, hypo_str),
                    file=output_file,
                )

            process_prediction(hypos[i][0])
            process_prediction(hypos[i][top_idx[i]])