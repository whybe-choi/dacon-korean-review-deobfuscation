import logging
import os
import random

import torch
from transformers import AutoTokenizer, HfArgumentParser, set_seed
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset, Dataset
from peft import AutoPeftModelForCausalLM

from arguments import DataArguments, ModelArguments
from load_model import get_model
from utils import create_dpo_prompt

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, DPOConfig))
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

    all_examples = load_dataset("csv", data_files="../data/train.csv", split="train")
    train_dataset = load_dataset("csv", data_files=data_args.train_data, split="train")

    def generate_prompt(examples, all_examples):
        prompts = []
        chosen_responses = []
        rejected_responses = []

        for idx, example in enumerate(zip(examples["input"], examples["chosen"], examples["rejected"])):
            input_text, chosen, rejected = example

            selected_indices = random.sample(range(len(all_examples)), 5)
            shots = [{"input": all_examples[i]["input"], "output": all_examples[i]["output"]} for i in selected_indices]

            prompt = create_dpo_prompt(shots, input_text)
            chosen = f"{chosen}<end_of_turn>" + tokenizer.eos_token
            rejected = f"{rejected}<end_of_turn>" + tokenizer.eos_token

            prompts.append(prompt)
            chosen_responses.append(chosen)
            rejected_responses.append(rejected)

        return Dataset.from_dict({"prompt": prompts, "chosen": chosen_responses, "rejected": rejected_responses})

    train_dataset = generate_prompt(train_dataset, all_examples)

    logging.info(f"Train dataset: {len(train_dataset)}")

    trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
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
