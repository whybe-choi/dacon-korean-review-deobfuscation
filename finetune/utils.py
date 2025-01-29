def create_prompt(examples, current_input, current_output):
    prompt = (
        "<bos>Your task is to transform the given obfuscated Korean review into a clear, correct, and natural-sounding Korean review that reflects its original meaning. "
        "Spacing and word length in the output must be restored to the same as in the input. Do not provide any description. Print only in Korean."
    )

    for example in examples:
        prompt += f"<start_of_turn>user\nTransform the following obfuscated review: {example['input']}<end_of_turn>\n<start_of_turn>model\n{example['output']}<end_of_turn>\n"

    prompt += f"<start_of_turn>user\nTransform the following obfuscated review: {current_input}<end_of_turn>\n<start_of_turn>model\n{current_output}<end_of_turn>"
    return prompt
