"""
Step 2: DPO 학습
- generate_preferences.py로 생성한 preference pair로 DPO 학습
- 학습 완료 후 → W8A8 양자화 → 벤치마크

Usage:
    # 기본 DPO
    python dpo_train.py --model ./base_model_2ssp_pruned_only --data ./preference_data

    # LoRA DPO (VRAM 절약)
    python dpo_train.py --model ./base_model_2ssp_pruned_only --data ./preference_data --lora

    # 학습률/에폭 조정
    python dpo_train.py --model ./base_model_2ssp_pruned_only --data ./preference_data --lr 5e-7 --epochs 2
"""

import argparse
import os
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, get_peft_model


def main():
    parser = argparse.ArgumentParser(description="DPO Fine-tuning for EXAONE-4.0-1.2B")
    parser.add_argument("--model", type=str, required=True, help="베이스 모델 경로")
    parser.add_argument("--data", type=str, required=True, help="preference data 경로")
    parser.add_argument("--output", type=str, default="./dpo_output", help="출력 경로")
    parser.add_argument("--lora", action="store_true", help="LoRA 사용 (VRAM 절약)")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lr", type=float, default=5e-7, help="학습률")
    parser.add_argument("--epochs", type=int, default=1, help="에폭 수")
    parser.add_argument("--batch-size", type=int, default=2, help="배치 사이즈")
    parser.add_argument("--grad-accum", type=int, default=8, help="gradient accumulation")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta (KL penalty)")
    parser.add_argument("--max-length", type=int, default=1024, help="최대 시퀀스 길이")
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    model_path = os.path.abspath(args.model)
    model_name = Path(model_path).name

    print(f"Model: {model_name}")
    print(f"Data: {args.data}")
    print(f"LoRA: {args.lora} (r={args.lora_r})" if args.lora else f"Full fine-tuning")
    print(f"LR: {args.lr}, Epochs: {args.epochs}, Beta: {args.beta}")
    print(f"Effective batch: {args.batch_size * args.grad_accum}")

    # 데이터 로드
    print("\n[1/4] Loading preference data...")
    dataset = load_from_disk(args.data)
    print(f"  {len(dataset)} preference pairs loaded")

    # Train/eval split
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # 모델 로드
    print("\n[2/4] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    # LoRA 설정
    if args.lora:
        print("\n[2.5/4] Applying LoRA...")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        ref_model = None  # trl이 LoRA에서는 자동으로 ref model 처리
    else:
        lora_config = None
        # Full fine-tune: ref model 필요
        print("  Loading reference model...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )

    # DPO 학습 설정
    print("\n[3/4] Setting up DPO training...")
    training_args = DPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        gradient_checkpointing=True,
        max_length=args.max_length,
        beta=args.beta,
        report_to="none",
        seed=42,
        dataloader_num_workers=4,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # 학습
    print("\n[4/4] Training...")
    trainer.train()

    # 저장
    print("\nSaving model...")
    final_path = os.path.join(args.output, "final")

    if args.lora:
        # LoRA merge → full model 저장
        print("  Merging LoRA weights...")
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(final_path)
    else:
        trainer.model.save_pretrained(final_path)

    tokenizer.save_pretrained(final_path)
    print(f"\nDone! Model saved to: {final_path}")
    print(f"\nNext steps:")
    print(f"  1. Quantize: python -c \"... W8A8 quantization ...\"")
    print(f"  2. Benchmark: python benchmark.py --model {final_path}")


if __name__ == "__main__":
    main()
