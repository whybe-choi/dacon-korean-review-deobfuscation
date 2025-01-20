from dataclasses import dataclass, field
from typing import Optional, List


def default_list() -> List[str]:
    return ["v_proj", "q_proj", "k_proj", "gate_proj", "down_proj", "o_proj", "up_proj"]


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override the default `torch.dtype` and load the model under this dtype.",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which attention implementation to use. You can run `--attn_implementation=flash_attention_2`, in "
            "which case you must install this manually by running `pip install flash-attn --no-build-isolation`."
        },
    )
    use_bnb: Optional[bool] = field(default=True, metadata={"help": "Whether to use BitsAndBytes."})
    use_lora: Optional[bool] = field(default=False, metadata={"help": "Whether to use LoRA."})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter."})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter."})
    lora_rank: Optional[int] = field(default=32, metadata={"help": "the lora rank parameter."})
    low_cpu_mem_usage: Optional[bool] = field(default=False, metadata={"help": "Whether to use low cpu memory usage."})
    cache_dir: Optional[str] = field(default="./LMs")
    token: Optional[str] = field(default="")
    peft_model_path: Optional[str] = field(default="")
    from_peft: Optional[str] = field(default=None)
