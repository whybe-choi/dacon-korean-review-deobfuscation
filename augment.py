import argparse
import random
import pandas as pd
import hgtk

CONSONANT_MAP = {
    "ㄱ": "ㅋ",
    "ㅋ": "ㄱ",
    "ㄲ": "ㄱ",
    "ㄴ": "ㄹ",
    "ㄷ": "ㅌ",
    "ㅌ": "ㄷ",
    "ㄸ": "ㄷ",
    "ㄹ": "ㄴ",
    "ㅂ": "ㅍ",
    "ㅍ": "ㅂ",
    "ㅃ": "ㅂ",
    "ㅅ": "ㅆ",
    "ㅆ": "ㅅ",
    "ㅈ": "ㅉ",
    "ㅉ": "ㅈ",
    "ㅊ": "ㅈ",
    "ㅎ": "ㅇ",
    "ㅇ": "ㅎ",
}

VOWEL_MAP = {
    "ㅏ": "ㅑ",
    "ㅑ": "ㅏ",
    "ㅓ": "ㅕ",
    "ㅕ": "ㅓ",
    "ㅗ": "ㅛ",
    "ㅛ": "ㅗ",
    "ㅜ": "ㅠ",
    "ㅠ": "ㅜ",
    "ㅡ": "ㅢ",
    "ㅢ": "ㅡ",
    "ㅐ": "ㅔ",
    "ㅔ": "ㅐ",
    "ㅒ": "ㅖ",
    "ㅖ": "ㅒ",
    "ㅘ": "ㅙ",
    "ㅙ": "ㅘ",
    "ㅝ": "ㅞ",
    "ㅞ": "ㅝ",
}

RANDOM_JONGSUNG = [
    "",
    "ㄱ",
    "ㄴ",
    "ㄷ",
    "ㄹ",
    "ㅁ",
    "ㅂ",
    "ㅅ",
    "ㅇ",
    "ㅈ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
    "ㄲ",
    "ㄵ",
    "ㄶ",
    "ㄺ",
    "ㄻ",
    "ㄼ",
    "ㄽ",
    "ㄾ",
    "ㄿ",
    "ㅀ",
    "ㅄ",
]


def obfuscate_text(text, error_prob=0.3):
    obfuscated_text = ""

    for char in text:
        try:
            cho, jung, jong = hgtk.letter.decompose(char)

            if random.random() < error_prob and cho in CONSONANT_MAP:
                cho = CONSONANT_MAP[cho]
            if random.random() < error_prob and jung in VOWEL_MAP:
                jung = VOWEL_MAP[jung]
            if random.random() < error_prob:
                jong = random.choice(RANDOM_JONGSUNG) if random.random() < 0.9 else ""
            if random.random() < error_prob * 0.5:
                cho = cho * 2

            obfuscated_text += hgtk.letter.compose(cho, jung, jong)

        except hgtk.exception.NotHangulException:
            obfuscated_text += char

    return obfuscated_text


def main():
    parser = argparse.ArgumentParser(description="Augment obfuscated Korean reviews.")
    parser.add_argument("--input_file", type=str, default="../data/train.csv")
    parser.add_argument("--output_file", type=str, default="../data/augmented_train.csv")
    parser.add_argument("--k", type=int, default=5, help="number of obfuscated sentences per original sentence")
    parser.add_argument("--error_prob", type=float, default=0.4, help="probability of obfuscation error")

    args = parser.parse_args()

    df = pd.read_csv(args.input_file, encoding="utf-8-sig")
    df_orig = df.copy()

    df["input"] = df["output"].apply(lambda x: [obfuscate_text(str(x), error_prob=args.error_prob) for _ in range(args.k)])

    df = df.explode("input")
    df = pd.concat([df_orig, df], ignore_index=True)

    df.to_csv(args.output_file, index=False, encoding="utf-8-sig")

    print(f"✅ {args.k}개의 난독화된 문장이 {args.output_file}에 저장되었습니다!")


if __name__ == "__main__":
    main()
