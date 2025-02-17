import argparse
import logging
from tqdm import tqdm
import random

import torch
import pandas as pd
from transformers import pipeline
from datasets import load_dataset


logger = logging.getLogger(__name__)


def create_prompt(train_dataset, input_text: str, n_shot=5) -> str:
    shots = train_dataset.select(random.sample(range(len(train_dataset)), n_shot))

    prompt = (
        "<bos>"
        "Your task is to transform the given obfuscated Korean review into a clear, correct, and natural-sounding Korean review "
        "that reflects its original meaning. Spacing and word length in the output must be restored to the same as in the input. "
        "Do not provide any description. Print only in Korean."
    )
    for shot in shots:
        prompt += (
            "<start_of_turn>user\n"
            f"Transform the following obfuscated review: {shot['input']}"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
            f"{shot['output']}"
            "<end_of_turn>\n"
        )
    prompt += (
        f"<start_of_turn>user\nTransform the following obfuscated review: {input_text}<end_of_turn>\n<start_of_turn>model\n"
    )

    return prompt


def main():
    parser = argparse.ArgumentParser(description="Inference for obfuscated Korean reviews.")
    parser.add_argument("--model_name_or_path", type=str, default="whybe-choi/ko-gemma-2-9b-it-5shot-dacon")
    parser.add_argument("--train_path", type=str, default="./data/train.csv")
    parser.add_argument("--test_path", type=str, default="./data/test.csv")
    parser.add_argument("--submission_path", type=str, default="./submissions/submission.csv")
    parser.add_argument("--n_shot", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--do_sample", type=bool, default=False)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )

    logging.info(f"Arguments: {args}")

    text_generator = pipeline(
        "text-generation",
        model=args.model_name_or_path,
        tokenizer=args.model_name_or_path,
        torch_dtype=torch.float16,
        device=0,
    )

    train_dataset = load_dataset("csv", data_files=args.train_path, split="train")
    test_dataset = load_dataset("csv", data_files=args.test_path, split="train")

    test_dataset = test_dataset.map(lambda x: {"prompt": create_prompt(train_dataset, x["input"], n_shot=args.n_shot)})

    results = []
    for idx, example in enumerate(test_dataset):
        result = text_generator(
            example["prompt"],
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
        )
        generated_text = result[0]["generated_text"][len(example["prompt"]) :]
        generated_text = generated_text[: len(example["input"])]
        generated_text = generated_text.replace("<end_of_turn>", "")
        generated_text = generated_text.replace("\n", "")
        results.append(generated_text)

        logging.info(f"Example {idx + 1}/{len(test_dataset)}")
        logging.info(f"Input  : {example['input']}")
        logging.info(f"Output : {generated_text}\n")

    logging.info(f"Saving results to {args.submission_path} ...")
    submission = pd.read_csv("./data/sample_submission.csv", encoding="utf-8-sig")
    submission["output"] = results
    submission.to_csv(args.submission_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
