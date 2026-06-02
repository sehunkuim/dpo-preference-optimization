"""
2-Stage Post-Pruning Recovery:
  Stage 1: Knowledge Distillation (teacher → student)
  Stage 2: DPO (짧고 정확한 응답 선호)

Usage:
    python distill_and_dpo.py --teacher /path/base --student /path/pruned --output /path/out
"""

import argparse
import os
import random
import re

import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# ─── Stage 1: Distillation ───

def run_distillation(teacher, student, tokenizer, train_data, args):
    """KL divergence distillation"""
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
        lr=args.lr, weight_decay=0.01,
    )
    total_steps = len(train_data) // args.batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    student.train()
    teacher.eval()

    indices = torch.randperm(len(train_data))
    pbar = tqdm(range(0, len(train_data), args.batch_size), desc="Distillation")
    step, total_loss = 0, 0

    for batch_start in pbar:
        batch_idx = indices[batch_start:batch_start + args.batch_size]
        texts = [train_data[int(i)]["text"] for i in batch_idx]
        encoded = tokenizer(texts, return_tensors="pt", truncation=True,
                           max_length=args.max_length, padding=True).to("cuda:0")

        with torch.no_grad():
            t_logits = teacher(input_ids=encoded.input_ids,
                              attention_mask=encoded.attention_mask).logits

        s_logits = student(input_ids=encoded.input_ids,
                          attention_mask=encoded.attention_mask).logits

        # Shift
        s = s_logits[:, :-1, :].contiguous()
        t = t_logits[:, :-1, :].contiguous()
        labels = encoded.input_ids[:, 1:].contiguous()

        # KL + CE
        temp = args.temperature
        kl = F.kl_div(
            F.log_softmax(s / temp, dim=-1),
            F.softmax(t / temp, dim=-1),
            reduction="batchmean"
        ) * (temp ** 2)
        ce = F.cross_entropy(s.view(-1, s.size(-1)), labels.view(-1),
                            ignore_index=tokenizer.pad_token_id or 0)
        loss = args.alpha * kl + (1 - args.alpha) * ce

        loss.backward()
        step += 1

        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        if step % 50 == 0:
            pbar.set_postfix(loss=f"{total_loss/step:.4f}")

    print(f"Distillation done: avg loss={total_loss/step:.4f}")


# ─── Stage 2: DPO Preference Generation + Training ───

def extract_answer(text):
    if not text: return None
    s = text.split("</think>")[-1] if "</think>" in text else text
    m = re.search(r"정답\s*(?:은|:|는|이)*\s*([A-D])", s, re.IGNORECASE)
    if m: return m.group(1).upper()
    p = re.findall(r"\b([A-D])\b", s)
    return p[-1].upper() if p else None


def generate_dpo_pairs(student, tokenizer, args):
    """Student 모델로 MCQ 응답 생성 → chosen(짧은 정답) / rejected(오답) pair"""
    from vllm import LLM, SamplingParams

    print("Loading student into vLLM for pair generation...")
    # student를 임시 저장 후 vLLM으로 로드
    tmp_dir = os.path.join(args.output, "_tmp_for_vllm")
    os.makedirs(tmp_dir, exist_ok=True)
    student.save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)

    # GPU 정리
    del student
    torch.cuda.empty_cache()

    llm = LLM(model=tmp_dir, tensor_parallel_size=1, gpu_memory_utilization=0.85,
              trust_remote_code=True, max_model_len=4096, disable_log_stats=True)

    # KMMLU train 데이터 로드
    print("Loading KMMLU train data...")
    all_questions = []
    subjects = ["Accounting", "Biology", "Computer-Science", "Economics", "Math",
                "Chemistry", "Education", "Korean-History", "Management", "Taxation",
                "Electrical-Engineering", "Criminal-Law", "Civil-Engineering",
                "Information-Technology", "Marketing"]
    for sub in subjects:
        try:
            ds = load_dataset("HAERAE-HUB/KMMLU", sub, split="train")
            for item in ds:
                all_questions.append({
                    "question": item["question"],
                    "A": item["A"], "B": item["B"], "C": item["C"], "D": item["D"],
                    "answer": chr(64 + int(item["answer"])),
                    "subject": sub,
                })
        except:
            pass

    random.shuffle(all_questions)
    questions = all_questions[:args.dpo_questions]
    print(f"Using {len(questions)} questions")

    # 응답 생성 (n=16, temp=1.0)
    sp = SamplingParams(temperature=1.0, top_p=0.9, max_tokens=256, n=args.dpo_n_samples)
    prompts = []
    for q in questions:
        opts = f"A. {q['A']}\nB. {q['B']}\nC. {q['C']}\nD. {q['D']}"
        text = f"질문: {q['question']}\n\n선택지:\n{opts}\n\n'정답: [알파벳]' 형식으로 답하세요."
        msg = [{"role": "system", "content": "정해진 양식으로 부연 설명 없이 답변하세요."},
               {"role": "user", "content": text}]
        prompts.append(tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))

    print(f"Generating {len(prompts)} × {args.dpo_n_samples} responses...")
    outputs = llm.generate(prompts, sp, use_tqdm=True)

    # Preference pair 구성
    pairs = []
    for q, output in zip(questions, outputs):
        target = q["answer"]
        correct, incorrect = [], []

        for comp in output.outputs:
            text = comp.text.strip()
            n_tok = len(comp.token_ids)
            pred = extract_answer(text)
            if pred == target:
                correct.append({"text": text, "tokens": n_tok})
            else:
                incorrect.append({"text": text, "tokens": n_tok})

        if correct and incorrect:
            # SimPER: 가장 짧은 정답 = chosen
            chosen = min(correct, key=lambda x: x["tokens"])
            rejected = random.choice(incorrect)

            opts = f"A. {q['A']}\nB. {q['B']}\nC. {q['C']}\nD. {q['D']}"
            prompt = f"질문: {q['question']}\n\n선택지:\n{opts}\n\n'정답: [알파벳]' 형식으로 답하세요."

            pairs.append({
                "prompt": prompt,
                "chosen": chosen["text"],
                "rejected": rejected["text"],
            })

    print(f"Generated {len(pairs)} preference pairs")

    del llm
    torch.cuda.empty_cache()

    # 임시 파일 정리
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return pairs


def run_dpo(student_path, tokenizer, pairs, args):
    """DPO training"""
    from trl import DPOConfig, DPOTrainer
    from peft import LoraConfig

    print(f"Loading student for DPO ({len(pairs)} pairs)...")
    student = AutoModelForCausalLM.from_pretrained(
        student_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="sdpa",
    )
    student.config.use_cache = False

    lora_config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )

    ds = Dataset.from_list(pairs)
    split = ds.train_test_split(test_size=0.05, seed=42)

    training_args = DPOConfig(
        output_dir=os.path.join(args.output, "dpo_checkpoints"),
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        gradient_checkpointing=True,
        max_length=512,
        beta=0.1,
        report_to="none",
        seed=42,
    )

    trainer = DPOTrainer(
        model=student,
        ref_model=None,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()

    # LoRA merge
    print("Merging LoRA...")
    merged = trainer.model.merge_and_unload()
    final_path = os.path.join(args.output, "final")
    os.makedirs(final_path, exist_ok=True)
    merged.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"DPO model saved: {final_path}")

    del trainer, merged
    torch.cuda.empty_cache()


# ─── Main ───

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=str, required=True)
    parser.add_argument("--student", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    # Distillation
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--num-samples", type=int, default=20000)
    parser.add_argument("--ft-layers", type=int, default=3)
    # DPO
    parser.add_argument("--dpo-questions", type=int, default=5000)
    parser.add_argument("--dpo-n-samples", type=int, default=16)
    parser.add_argument("--skip-distill", action="store_true")
    parser.add_argument("--skip-dpo", action="store_true")
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.output, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    distill_path = os.path.join(args.output, "distilled")

    # ─── Stage 1: Distillation ───
    if not args.skip_distill:
        print("=" * 60)
        print("  Stage 1: Knowledge Distillation")
        print("=" * 60)

        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher, torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="sdpa",
        ).cuda().eval()
        for p in teacher.parameters():
            p.requires_grad = False

        student = AutoModelForCausalLM.from_pretrained(
            args.student, torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="sdpa",
        ).cuda()

        # Freeze → unfreeze last N + lm_head
        for p in student.parameters():
            p.requires_grad = False
        for p in student.lm_head.parameters():
            p.requires_grad = True
        n_layers = student.config.num_hidden_layers
        for i in range(n_layers - args.ft_layers, n_layers):
            for p in student.model.layers[i].parameters():
                p.requires_grad = True

        trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
        total = sum(p.numel() for p in student.parameters())
        print(f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({trainable/total*100:.1f}%)")

        # 다양한 데이터 로드
        print(f"Loading diverse data ({args.num_samples} samples)...")
        all_texts = []

        # MANTA-1M (60%)
        manta = load_dataset("LGAI-EXAONE/MANTA-1M", split=f"train[:{int(args.num_samples*0.6)}]")
        for item in manta:
            all_texts.append({"text": tokenizer.apply_chat_template(
                item["conversations"], tokenize=False, add_generation_prompt=False)})

        # Orca (20%)
        try:
            orca = load_dataset("Open-Orca/OpenOrca", split=f"train[:{int(args.num_samples*0.2)}]")
            for item in orca:
                msg = [{"role": "system", "content": item.get("system_prompt", "")},
                       {"role": "user", "content": item["question"]},
                       {"role": "assistant", "content": item["response"]}]
                all_texts.append({"text": tokenizer.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=False)})
        except Exception as e:
            print(f"  Orca failed: {e}, using more MANTA")
            extra = load_dataset("LGAI-EXAONE/MANTA-1M",
                                split=f"train[{int(args.num_samples*0.6)}:{int(args.num_samples*0.8)}]")
            for item in extra:
                all_texts.append({"text": tokenizer.apply_chat_template(
                    item["conversations"], tokenize=False, add_generation_prompt=False)})

        # Ko instruction (20%)
        try:
            ko = load_dataset("heegyu/ko-chatgpt-alpaca", split=f"train[:{int(args.num_samples*0.2)}]")
            for item in ko:
                msg = [{"role": "user", "content": item["instruction"]},
                       {"role": "assistant", "content": item["output"]}]
                all_texts.append({"text": tokenizer.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=False)})
        except Exception as e:
            print(f"  Ko data failed: {e}, using more MANTA")
            extra = load_dataset("LGAI-EXAONE/MANTA-1M",
                                split=f"train[{int(args.num_samples*0.8)}:{args.num_samples}]")
            for item in extra:
                all_texts.append({"text": tokenizer.apply_chat_template(
                    item["conversations"], tokenize=False, add_generation_prompt=False)})

        random.shuffle(all_texts)
        print(f"  Total: {len(all_texts)} samples")

        student.gradient_checkpointing_enable()
        run_distillation(teacher, student, tokenizer, all_texts, args)
        student.gradient_checkpointing_disable()

        # 저장
        os.makedirs(distill_path, exist_ok=True)
        student.save_pretrained(distill_path)
        tokenizer.save_pretrained(distill_path)
        print(f"Distilled model saved: {distill_path}")

        del teacher, student
        torch.cuda.empty_cache()
    else:
        print("[SKIP] Distillation")

    # ─── Stage 2: DPO ───
    if not args.skip_dpo:
        print("\n" + "=" * 60)
        print("  Stage 2: DPO (Short-Response Preference)")
        print("=" * 60)

        # Distilled model로 preference pair 생성
        student_for_gen = AutoModelForCausalLM.from_pretrained(
            distill_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
        pairs = generate_dpo_pairs(student_for_gen, tokenizer, args)

        if len(pairs) < 100:
            print(f"Too few pairs ({len(pairs)}), skipping DPO")
        else:
            run_dpo(distill_path, tokenizer, pairs, args)
    else:
        print("[SKIP] DPO")

    print("\n" + "=" * 60)
    print("  ALL DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
