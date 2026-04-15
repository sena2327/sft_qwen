import argparse
import inspect
import os
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer

try:
    from trl import DPOConfig
except Exception:  # pragma: no cover
    DPOConfig = None

from utils.janome_tokenizer import JanomeRougeTokenizer

BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
REF_MODEL = "Qwen/Qwen3-0.6B-Base"
TRAIN_FILE = "data/train.jsonl"
VALIDATION_FILE = "data/validation.jsonl"
SYSTEM_PROMPT_FILE = "system_prompt.txt"
OUTPUT_DIR = "dpo_output"


def build_prompt(system_prompt: str, text: str) -> str:
    return f"{system_prompt}\n{text}\n答え:\n"


def load_system_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def generate_candidates(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    num_candidates: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(
        model.device
    )
    prompt_len = int(inputs["attention_mask"].sum(dim=1)[0].item())
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        num_return_sequences=num_candidates,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    candidates = []
    for out in outputs:
        pred_ids = out[prompt_len:]
        candidates.append(tokenizer.decode(pred_ids, skip_special_tokens=True).strip())
    return candidates


def choose_pair_by_rouge(
    scorer,
    reference: str,
    candidates: List[str],
    min_margin: float,
) -> Optional[Tuple[str, str]]:
    if len(candidates) < 2:
        return None
    scored = []
    for c in candidates:
        score = scorer.score(reference, c)["rougeL"].fmeasure
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, chosen = scored[0]
    worst_score, rejected = scored[-1]
    if (best_score - worst_score) < min_margin:
        return None
    if chosen == rejected:
        return None
    return chosen, rejected


def build_preference_dataset(
    model,
    tokenizer,
    split_ds,
    system_prompt: str,
    max_samples: int,
    max_new_tokens: int,
    num_candidates: int,
    temperature: float,
    top_p: float,
    min_margin: float,
) -> Dataset:
    custom_tokenizer = JanomeRougeTokenizer(use_stemmer=True)
    scorer = rouge_scorer.RougeScorer(
        ["rougeL"], use_stemmer=False, tokenizer=custom_tokenizer
    )

    records: List[Dict[str, str]] = []
    limit = min(len(split_ds), max_samples) if max_samples > 0 else len(split_ds)
    for i in range(limit):
        rec = split_ds[i]
        prompt = build_prompt(system_prompt, rec["text"])
        reference = rec["target"].strip()
        candidates = generate_candidates(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            num_candidates=num_candidates,
            temperature=temperature,
            top_p=top_p,
        )
        pair = choose_pair_by_rouge(
            scorer=scorer,
            reference=reference,
            candidates=candidates,
            min_margin=min_margin,
        )
        if pair is None:
            continue
        chosen, rejected = pair
        records.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    if not records:
        raise RuntimeError(
            "No preference pairs were created. Try more samples/candidates or lower min-margin."
        )
    return Dataset.from_list(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ROUGEに基づく選好データを構築してDPO学習する"
    )
    parser.add_argument("--policy-model", type=str, default=BASE_MODEL)
    parser.add_argument("--ref-model", type=str, default=REF_MODEL)
    parser.add_argument("--train-file", type=str, default=TRAIN_FILE)
    parser.add_argument("--validation-file", type=str, default=VALIDATION_FILE)
    parser.add_argument("--system-prompt-file", type=str, default=SYSTEM_PROMPT_FILE)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)

    parser.add_argument("--max-train-samples", type=int, default=2000)
    parser.add_argument("--max-eval-samples", type=int, default=300)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-margin", type=float, default=0.01)

    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument("--report-to", type=str, default="none")
    parser.add_argument("--run-name", type=str, default="qwen-dpo-rouge")
    args = parser.parse_args()

    if args.batch_size <= 0 or args.grad_accum <= 0:
        raise ValueError("batch-size and grad-accum must be > 0")

    ds = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.validation_file},
    )
    system_prompt = load_system_prompt(args.system_prompt_file)

    tokenizer = AutoTokenizer.from_pretrained(args.policy_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Pair generation model (sampling) with current policy.
    pair_model = AutoModelForCausalLM.from_pretrained(
        args.policy_model,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    pair_model.eval()

    ##Dataset
    train_pref = build_preference_dataset(
        model=pair_model,
        tokenizer=tokenizer,
        split_ds=ds["train"],
        system_prompt=system_prompt,
        max_samples=args.max_train_samples,
        max_new_tokens=args.max_new_tokens,
        num_candidates=args.num_candidates,
        temperature=args.temperature,
        top_p=args.top_p,
        min_margin=args.min_margin,
    )
    eval_pref = build_preference_dataset(
        model=pair_model,
        tokenizer=tokenizer,
        split_ds=ds["validation"],
        system_prompt=system_prompt,
        max_samples=args.max_eval_samples,
        max_new_tokens=args.max_new_tokens,
        num_candidates=args.num_candidates,
        temperature=args.temperature,
        top_p=args.top_p,
        min_margin=args.min_margin,
    )
    del pair_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        args.policy_model,
        dtype=torch.float32,
        trust_remote_code=True,
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.ref_model,
        dtype=torch.float32,
        trust_remote_code=True,
    )

    train_args_common = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to=[] if args.report_to == "none" else [args.report_to],
        run_name=args.run_name,
        fp16=torch.cuda.is_available(),
    )

    if DPOConfig is not None:
        dpo_args = DPOConfig(
            beta=args.beta,
            max_length=args.max_length,
            max_prompt_length=args.max_prompt_length,
            **train_args_common,
        )
    else:
        dpo_args = TrainingArguments(**train_args_common)

    trainer_kwargs = dict(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=train_pref,
        eval_dataset=eval_pref,
        beta=args.beta,
    )

    sig = inspect.signature(DPOTrainer.__init__).parameters
    if "processing_class" in sig:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig:
        trainer_kwargs["tokenizer"] = tokenizer

    if "max_length" in sig:
        trainer_kwargs["max_length"] = args.max_length
    if "max_prompt_length" in sig:
        trainer_kwargs["max_prompt_length"] = args.max_prompt_length

    trainer = DPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    train_pref.to_json(os.path.join(args.output_dir, "dpo_train_preferences.jsonl"))
    eval_pref.to_json(os.path.join(args.output_dir, "dpo_eval_preferences.jsonl"))
    print(f"Saved preference datasets and model to: {args.output_dir}")


if __name__ == "__main__":
    main()
