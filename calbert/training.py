import argparse
import logging
import os
import warnings
import random
import pickle
import re
import shutil
import glob
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import wandb

from transformers import (
    AlbertConfig,
    AlbertForMaskedLM,
    AdamW,
    get_linear_schedule_with_warmup,
)

from .tokenizer import CalbertTokenizer
from .dataset import CalbertDataset
from .utils import normalize_path

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

warnings.filterwarnings("ignore", category=DeprecationWarning)

log = logging.getLogger(__name__)


def arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ALBERT")
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        required=True,
        help="The folder where ca.bpe.VOCABSIZE-vocab.json and ca.bpe.VOCABSIZE-merges.txt are",
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="Where the {train|vaild}.VOCABSIZE.*.npy files are.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        type=Path,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--tensorboard-dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the tensorboard logs will be written to.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=128,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=128,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )

    parser.add_argument(
        "--subset",
        default=1.0,
        type=float,
        help="Percentage of dataset to use",
    )

    parser.add_argument(
        "--no_cuda", default=False, help="Avoid using CUDA when available"
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O3",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )

    parser.add_argument(
        "--logging_steps", type=int, default=200, help="Log every X updates steps."
    )
    parser.add_argument(
        "--eval_steps", type=int, default=24000, help="Evaluate every X steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=50,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )

    parser.add_argument(
        "--wandb",
        type=bool,
        default=False,
        help="Log training to Weights and Biases",
    )

    parser.add_argument(
        "--local-rank",
        type=int,
        default=-1,
        help="Node number -- set by Pytorch in distributed training",
    )
    return parser


def evaluate(args, cfg, model, tokenizer, device, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = str(args.out_dir)

    valid_dataset = CalbertDataset(dataset_dir=args.dataset_dir, split='valid', max_seq_length=cfg.training.max_seq_length, max_vocab_size=cfg.vocab.max_size, subset=args.subset)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(valid_dataset)
    eval_dataloader = DataLoader(
        valid_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    if args.local_rank in [-1, 0]:
        log.info("***** Running evaluation {} *****".format(prefix))
        log.info("  Num examples = %d", len(valid_dataset))
        log.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, special_tokens_masks, attention_masks, token_type_ids = batch.permute(1, 0, 2)
        inputs, labels = mask_tokens(inputs, special_tokens_masks, tokenizer, cfg)
        inputs = inputs.to(device)
        labels = labels.to(device)
        attention_masks = attention_masks.to(device)
        token_type_ids = token_type_ids.to(device)

        with torch.no_grad():
            outputs = model(
                inputs,
                masked_lm_labels=labels,
                attention_mask=attention_masks,
                token_type_ids=token_type_ids,
            )
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity}

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        log.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            log.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def mask_tokens(
    inputs: torch.Tensor,
    special_tokens_mask: torch.Tensor,
    tokenizer: CalbertTokenizer,
    cfg,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, cfg.training.masked_lm_prob)
    probability_matrix.masked_fill_(
        special_tokens_mask.bool(), value=0.0
    )
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[
        ~masked_indices
    ] = (
        -1
    )  # We only compute loss on masked tokens TODO: change to -100 once the change is merged in tranformers

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    )
    inputs[indices_replaced] = tokenizer.mask_token_id

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )

    le = len(tokenizer)
    random_words = torch.randint(le, labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(
        os.path.join(str(args.out_dir), "{}-*".format(checkpoint_prefix))
    )
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append(
                    (int(regex_match.groups()[0]), path)
                )

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(
        0, len(checkpoints_sorted) - args.save_total_limit
    )
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        log.info(
            "Deleting older checkpoint [{}] due to args.save_total_limit".format(
                checkpoint
            )
        )
        shutil.rmtree(checkpoint)


def _train(args, cfg, dataset, model, tokenizer, device):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(args.tensorboard_dir)

    train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    )

    train_dataloader = DataLoader(
        dataset, sampler=train_sampler, batch_size=train_batch_size
    )

    gradient_accumulation_steps = 1

    if cfg.training.num_train_steps > 0:
        t_total = cfg.training.num_train_steps
        cfg.training.epochs = (
            t_total // (len(train_dataloader) // gradient_accumulation_steps) + 1
        )
    else:
        t_total = (
            len(train_dataloader) // gradient_accumulation_steps * cfg.training.epochs
        )

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": cfg.training.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=cfg.training.learning_rate,
        eps=cfg.training.adam_epsilon,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.training.num_warmup_steps,
        num_training_steps=t_total,
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(str(args.out_dir), "optimizer.pt")) and os.path.isfile(
        os.path.join(str(args.out_dir), "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(
            torch.load(os.path.join(str(args.out_dir), "optimizer.pt"))
        )
        scheduler.load_state_dict(
            torch.load(os.path.join(str(args.out_dir), "scheduler.pt"))
        )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    if args.wandb and args.local_rank in [-1, 0]:
        wandb.config.train_batch_size = train_batch_size
        wandb.config.subset = args.subset
        wandb.config.gpus = torch.cuda.device_count()
    # wandb.watch(model, log='all')

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    log.info(f"***** Running training for {args.model_name}*****")
    log.info("  Num examples = %d", len(dataset))
    log.info("  Num Epochs = %d", cfg.training.epochs)
    log.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    log.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        train_batch_size
        * gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    log.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
    log.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    if os.path.exists(str(args.out_dir) + "/" + args.model_name):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (
                len(train_dataloader) // gradient_accumulation_steps
            )
            steps_trained_in_current_epoch = global_step % (
                len(train_dataloader) // gradient_accumulation_steps
            )

            log.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            log.info("  Continuing training from epoch %d", epochs_trained)
            log.info("  Continuing training from global step %d", global_step)
            log.info(
                "  Will skip the first %d steps in the first epoch",
                steps_trained_in_current_epoch,
            )
        except ValueError:
            log.info("  Starting training from scratch.")

    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(
        epochs_trained,
        int(cfg.training.epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )
    set_seed(args, cfg)  # Added here for reproducibility

    for _ in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        )
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            inputs, special_tokens_masks, attention_masks, token_type_ids = batch.permute(1, 0, 2)
            inputs, labels = mask_tokens(inputs, special_tokens_masks, tokenizer, cfg)
            inputs = inputs.to(device)
            labels = labels.to(device)
            attention_masks = attention_masks.to(device)
            token_type_ids = token_type_ids.to(device)

            model.train()
            outputs = model(
                inputs,
                masked_lm_labels=labels,
                attention_mask=attention_masks,
                token_type_ids=token_type_ids,
            )
            loss = outputs[
                0
            ]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), cfg.training.max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.training.max_grad_norm
                    )
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    # Log metrics
                    if (
                        args.local_rank in [-1, 0]
                        and args.eval_steps > 0
                        and global_step % args.eval_steps == 0
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, cfg, model, tokenizer, device)
                        for key, value in results.items():
                            tb_writer.add_scalar(
                                "eval_{}".format(key), value, global_step
                            )
                            if args.wandb:
                                wandb.log({f"val_{key}": value})
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        "loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_step,
                    )
                    if args.wandb:
                        wandb.log(
                            {"tr_loss": (tr_loss - logging_loss) / args.logging_steps}
                        )
                    logging_loss = tr_loss

                if (
                    args.local_rank in [-1, 0]
                    and args.save_steps > 0
                    and global_step % args.save_steps == 0
                ):
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(
                        str(args.out_dir), "{}-{}".format(checkpoint_prefix, global_step)
                    )
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    # tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    log.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(
                        optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                    )
                    torch.save(
                        scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                    )
                    log.info("Saving optimizer and scheduler states to %s", output_dir)

            if (
                cfg.training.num_train_steps > 0
                and global_step > cfg.training.num_train_steps
            ):
                epoch_iterator.close()
                break
        if (
            cfg.training.num_train_steps > 0
            and global_step > cfg.training.num_train_steps
        ):
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def set_seed(args, cfg):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(cfg.seed)


def train(args, cfg):
    if args.wandb and args.local_rank in [-1, 0]:
        wandb.init(project="calbert", sync_tensorboard=True)

    args.tokenizer_dir = normalize_path(args.tokenizer_dir)
    args.dataset_dir = normalize_path(args.dataset_dir)
    args.tensorboard_dir = normalize_path(args.tensorboard_dir)
    args.out_dir = normalize_path(args.out_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    args.model_name = (
        f"calbert-{cfg.model.name}-{'uncased' if cfg.vocab.lowercase else 'cased'}"
    )
    # Setup CUDA, GPU & distributed training
    device = None
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    if args.local_rank in [-1, 0]:
        log.info(f"Pretraining ALBERT: {args}")
        log.warning(
            "Device: %s, gpus: %s, distributed training: %s, 16-bits training: %s",
            device,
            torch.cuda.device_count(),
            bool(args.local_rank != -1),
            args.fp16,
        )

    tokenizer = CalbertTokenizer.from_dir(args.tokenizer_dir, max_seq_length=cfg.training.max_seq_length)

    if args.wandb and args.local_rank in [-1, 0]:
        c = dict(cfg.training)
        for k in c.keys():
            wandb.config[k] = c[k]

    c = dict(cfg.model)
    if args.wandb and args.local_rank in [-1, 0]:
        for k in c.keys():
            wandb.config[k] = c[k]

    config = AlbertConfig(**c)

    model = AlbertForMaskedLM(config)

    # Hack until this is released: https://github.com/huggingface/transformers/commit/100e3b6f2133074bf746a78eb4c3a0ad3e939b5f#diff-961191c6a9886609f732e878f0a1fa30
    # Otherwise the resizing doesn't apply for some layers
    model.predictions.decoder.bias = model.predictions.bias

    model_to_resize = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model = model.to(device)

    train_dataset = CalbertDataset(dataset_dir=args.dataset_dir, split='train', max_seq_length=cfg.training.max_seq_length, max_vocab_size=cfg.vocab.max_size, subset=args.subset)

    if args.local_rank in [-1, 0]:
        log.info("Loaded %i examples", len(train_dataset))

    # if args.local_rank == 0:
    #    torch.distributed.barrier()

    global_step, tr_loss = _train(args, cfg, train_dataset, model, tokenizer, device)

    if args.local_rank in [-1, 0]:
        log.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        # Create output directory if needed
        if not os.path.exists(str(args.out_dir)) and args.local_rank in [-1, 0]:
            os.makedirs(str(args.out_dir))

        log.info("Saving model checkpoint to %s", str(args.out_dir))
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(str(args.out_dir))

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(str(args.out_dir), "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AlbertForMaskedLM.from_pretrained(str(args.out_dir))
        model.to(device)

    return model
