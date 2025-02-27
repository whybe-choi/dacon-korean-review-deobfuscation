import argparse
import logging
import random

import torch
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams


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
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )

    logging.info(f"Arguments: {args}")

    llm = LLM(
        model=args.model_name_or_path,
        dtype="float16",
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature if args.temperature is not None else 0.0,
        top_p=args.top_p if args.top_p is not None else 1.0,
        top_k=args.top_k if args.top_k is not None else -1,
        use_beam_search=args.num_beams > 1,
        best_of=args.num_beams,
        n=1,
        stop=["user"],
    )

    train_dataset = load_dataset("csv", data_files=args.train_path, split="train")
    test_dataset = load_dataset("csv", data_files=args.test_path, split="train")
    test_dataset = test_dataset.map(lambda x: {"prompt": create_prompt(train_dataset, x["input"], n_shot=args.n_shot)})

    prompts = [example["prompt"] for example in test_dataset]
    
    logging.info("Starting batch inference with vLLM...")
    outputs = llm.generate(prompts, sampling_params)
    
    results = []
    for idx, (example, output) in enumerate(zip(test_dataset, outputs)):
        generated_text = output.outputs[0].text
        generated_text = generated_text.replace("\n", "")
        results.append(generated_text)

        logging.info(f"Example {idx + 1}/{len(test_dataset)}")
        logging.info(f"Input  : {example['input']}")
        logging.info(f"Output : {generated_text}\n")

    logging.info(f"Saving results to {args.submission_path} ...")
    submission = pd.read_csv(args.test_path, encoding="utf-8-sig")
    submission["output"] = results
    submission.to_csv(args.submission_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()