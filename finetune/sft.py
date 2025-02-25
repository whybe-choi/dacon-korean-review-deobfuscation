import logging
import os
import random

import torch
from transformers import AutoTokenizer, HfArgumentParser, set_seed
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM

from arguments import DataArguments, ModelArguments
from load_model import get_model
from utils import create_sft_prompt

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, SFTConfig))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    set_seed(training_args.seed)

    model = get_model(model_args)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, token=model_args.token, cache_dir=model_args.cache_dir
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if training_args.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    train_dataset = load_dataset("csv", data_files=data_args.train_data, split="train")

    def prompt_formatting_func(examples):
        texts = []

        all_indices = list(range(len(train_dataset)))
        inputs = examples["input"]
        outputs = examples["output"]

        for idx, (input, output) in enumerate(zip(inputs, outputs)):
            possible_indices = [i for i in all_indices if i != idx]
            selected_indices = random.sample(possible_indices, 5)
            shots = [train_dataset[i] for i in selected_indices]

            prompt = create_sft_prompt(shots, input, output) + tokenizer.eos_token
            texts.append(prompt)

        return texts

    logging.info(f"Train dataset: {len(train_dataset)}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        formatting_func=prompt_formatting_func,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)

    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)

    del model
    torch.cuda.empty_cache()

    model = AutoPeftModelForCausalLM.from_pretrained(output_dir, torch_dtype=model_args.torch_dtype)
    model = model.merge_and_unload()

    output_merged_dir = os.path.join(training_args.output_dir, "final_merged_checkpoint")
    model.save_pretrained(output_merged_dir, safe_serialize=True)

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_merged_dir)


if __name__ == "__main__":
    main()
