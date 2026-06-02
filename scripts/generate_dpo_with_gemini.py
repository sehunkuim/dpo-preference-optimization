"""
Gemini API로 고퀄리티 DPO Preference Pair 생성
1. Pruned 모델로 reasoning 응답 생성
2. Gemini로 reasoning quality 평가
3. chosen (좋은 reasoning + 정답) / rejected (나쁜 reasoning or 오답) pair 구성

Usage:
    python generate_dpo_with_gemini.py --model ./pruned_29L --output ./dpo_data --api-key YOUR_KEY
"""

import argparse
import json
import os
import random
import re
import time

import google.generativeai as genai
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm


def extract_answer(text):
    if not text: return None
    s = text.split("</think>")[-1] if "</think>" in text else text
    m = re.search(r"정답\s*(?:은|:|는|이)*\s*([A-D])", s, re.IGNORECASE)
    if m: return m.group(1).upper()
    p = re.findall(r"\b([A-D])\b", s)
    return p[-1].upper() if p else None


def load_mcq_data(num_questions):
    """KMMLU + MMLU train 데이터 로드"""
    all_questions = []

    # KMMLU
    kmmlu_subs = ["Accounting", "Biology", "Computer-Science", "Economics", "Math",
                  "Chemistry", "Education", "Korean-History", "Management", "Taxation",
                  "Electrical-Engineering", "Criminal-Law", "Civil-Engineering",
                  "Information-Technology", "Marketing", "Electronics-Engineering",
                  "Environmental-Science", "Health", "Construction", "Patent"]
    for sub in kmmlu_subs:
        try:
            ds = load_dataset("HAERAE-HUB/KMMLU", sub, split="train")
            for item in ds:
                all_questions.append({
                    "question": item["question"],
                    "options": f"A. {item['A']}\nB. {item['B']}\nC. {item['C']}\nD. {item['D']}",
                    "answer": chr(64 + int(item["answer"])),
                    "subject": sub, "lang": "ko",
                })
        except: pass

    # MMLU (영어)
    mmlu_subs = ["college_chemistry", "college_physics", "college_mathematics",
                 "high_school_chemistry", "high_school_physics", "econometrics"]
    for sub in mmlu_subs:
        try:
            ds = load_dataset("cais/mmlu", sub, split="test")
            for item in ds:
                c = item["choices"]
                all_questions.append({
                    "question": item["question"],
                    "options": f"A. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}",
                    "answer": chr(65 + int(item["answer"])),
                    "subject": sub, "lang": "en",
                })
        except: pass

    random.shuffle(all_questions)
    return all_questions[:num_questions]


def generate_responses(model_path, questions, tokenizer, n_samples=8):
    """vLLM으로 reasoning 응답 생성"""
    print(f"Loading model for generation...")
    llm = LLM(model=model_path, tensor_parallel_size=1, gpu_memory_utilization=0.85,
              trust_remote_code=True, max_model_len=4096, disable_log_stats=True)

    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1024, n=n_samples)

    prompts = []
    for q in questions:
        text = f"질문: {q['question']}\n\n선택지:\n{q['options']}\n\n단계별로 생각한 후 '정답: [알파벳]' 형식으로 답하세요."
        msg = [{"role": "system", "content": "단계별로 추론한 후 정해진 양식으로 답변하세요."},
               {"role": "user", "content": text}]
        prompts.append(tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))

    print(f"Generating {len(prompts)} × {n_samples} responses...")
    outputs = llm.generate(prompts, sp, use_tqdm=True)

    results = []
    for q, output in zip(questions, outputs):
        responses = []
        for comp in output.outputs:
            text = comp.text.strip()
            pred = extract_answer(text)
            responses.append({
                "text": text,
                "pred": pred,
                "correct": pred == q["answer"],
                "tokens": len(comp.token_ids),
            })
        results.append({"question": q, "responses": responses})

    del llm
    import torch; torch.cuda.empty_cache()
    return results


def evaluate_with_gemini(results, api_key, batch_size=5):
    """Gemini로 reasoning quality 평가"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    pairs = []
    errors = 0

    for item in tqdm(results, desc="Gemini evaluation"):
        q = item["question"]
        responses = item["responses"]

        correct_resps = [r for r in responses if r["correct"]]
        incorrect_resps = [r for r in responses if not r["correct"]]

        if not correct_resps or not incorrect_resps:
            continue

        # Gemini에게 정답 응답들 중 best를 골라달라고 요청
        if len(correct_resps) > 1:
            eval_prompt = f"""다음 객관식 문제에 대한 여러 풀이가 있습니다. 가장 논리적이고 간결한 풀이를 골라주세요.

문제: {q['question']}
선택지:
{q['options']}
정답: {q['answer']}

"""
            for i, r in enumerate(correct_resps[:4]):
                eval_prompt += f"--- 풀이 {i+1} ---\n{r['text'][:500]}\n\n"

            eval_prompt += "가장 좋은 풀이 번호만 답하세요 (숫자만): "

            try:
                response = model.generate_content(eval_prompt)
                best_idx_text = response.text.strip()
                best_idx = int(re.search(r'\d+', best_idx_text).group()) - 1
                best_idx = max(0, min(best_idx, len(correct_resps) - 1))
                chosen = correct_resps[best_idx]
            except Exception as e:
                errors += 1
                chosen = min(correct_resps, key=lambda x: x["tokens"])
                time.sleep(0.5)
        else:
            chosen = correct_resps[0]

        # Rejected: 가장 긴 오답 (장황하고 틀린 것)
        rejected = max(incorrect_resps, key=lambda x: x["tokens"])

        prompt_text = f"질문: {q['question']}\n\n선택지:\n{q['options']}\n\n단계별로 생각한 후 '정답: [알파벳]' 형식으로 답하세요."

        pairs.append({
            "prompt": prompt_text,
            "chosen": chosen["text"],
            "rejected": rejected["text"],
            "subject": q["subject"],
            "answer": q["answer"],
            "chosen_tokens": chosen["tokens"],
            "rejected_tokens": rejected["tokens"],
        })

        # Rate limit
        time.sleep(0.3)

    print(f"Generated {len(pairs)} pairs (Gemini errors: {errors})")
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="응답 생성할 모델")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--api-key", type=str, required=True, help="Gemini API key")
    parser.add_argument("--num-questions", type=int, default=5000)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # 1. 데이터 로드
    print("[1/3] Loading MCQ data...")
    questions = load_mcq_data(args.num_questions)
    print(f"  {len(questions)} questions loaded")

    # 2. 응답 생성
    print("\n[2/3] Generating responses...")
    results = generate_responses(args.model, questions, tokenizer, args.n_samples)

    # 3. Gemini 평가
    print("\n[3/3] Evaluating with Gemini...")
    pairs = evaluate_with_gemini(results, args.api_key)

    # 저장
    os.makedirs(args.output, exist_ok=True)
    ds = Dataset.from_list(pairs)
    ds.save_to_disk(args.output)

    with open(os.path.join(args.output, "sample.json"), "w", encoding="utf-8") as f:
        json.dump(pairs[:5], f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(pairs)} pairs to {args.output}")


if __name__ == "__main__":
    main()
