# dacon-korean-review-deobfuscation
<img width="1194" alt="Image" src="https://github.com/user-attachments/assets/f6552650-d7b1-423d-8c3b-1c54308c5763" />

## Results
|Name|Type|Performance|Rank|
|---|---|---|---|
|**[난독화된 한글 리뷰 복원 AI 경진대회](https://dacon.io/competitions/official/236446/overview/description)**|NLP, LLM|||

## Environment

```bash
conda create -n dacon python=3.10
conda activate dacon
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Supervised Fine-tuning (SFT)
```bash
CURRENT_TIME=$(date "+%Y-%m-%d_%H-%M-%S")

cd ./finetune

torchrun --nproc_per_node 1 \
sft.py \
--output_dir ./output \
--model_name_or_path rtzr/ko-gemma-2-9b-it \
--torch_dtype float16 \
--max_seq_length 1024 \
--train_data ../data/train.csv \
--learning_rate 3e-4 \
--num_train_epochs 5 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 8 \
--logging_steps 10 \
--save_strategy epoch \
--attn_implementation eager \
--warmup_ratio 0.1 \
--ddp_find_unused_parameters False \
--gradient_checkpointing \
--deepspeed ../stage1.json \
--fp16 \
--cache_dir ./LMs \
--token .. \
--report_to wandb \
--run_name rtzr-gemma-${CURRENT_TIME} \
```

## Inference
```bash
cd ./inference

python inference_vllm.py \
    --model_name_or_path ojoo/ko-gemma-2-9b-it-deobfuscation \
    --train_path ../data/train.csv \
    --test_path ../data/test.csv \
    --submission_path ../submissions/submission_total.csv \
    --n_shot 4 \
    --num_beams 1 \
    --max_new_tokens 1024
```
```bash
cd ./inference

python inference_vllm.py \
    --model_name_or_path whybe-choi/ko-gemma-2-9b-it-sft-dacon \
    --train_path ../data/train.csv \
    --test_path ../data/test_sentences.csv \
    --submission_path ../submissions/submission_sentences.csv \
    --n_shot 5 \
    --num_beams 5 \
    --max_new_tokens 1024
```