def create_prompt(input, output):
    prompt = (
        "<bos><start_of_turn>user\n"
        "Your task is to transform the given obfuscated Korean review into a clear, correct, and natural-sounding Korean review that reflects its original meaning.\n"
        f"###Input: {input}<end_of_turn>"
        "<start_of_turn>model\n"
        f"###Output: {output}<end_of_turn>"
    )
    return prompt
