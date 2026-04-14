import argparse
import inspect
import os
import tempfile
from typing import Dict, List

import numpy as np
import torch
from datasets import load_dataset
from rouge_score import rouge_scorer
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
)
from trl import SFTTrainer
from utils.janome_tokenizer import JanomeRougeTokenizer

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
except Exception:  # pragma: no cover - runtime dependency check
    LoraConfig = None
    PeftModel = None
    TaskType = None
    get_peft_model = None

try:
    from vllm import LLM, SamplingParams
except Exception:  # pragma: no cover - runtime dependency check
    LLM = None
    SamplingParams = None

BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
TRAIN_FILE = "data/train.jsonl"
VALIDATION_FILE = "data/validation.jsonl"
OUTPUT_DIR = "sft_output_lora"
SYSTEM_PROMPT_FILE = "system_prompt.txt"
EVAL_MAX_NEW_TOKENS = 128


def format_record(record: Dict[str, str], system_prompt: str) -> Dict[str, str]:
    prompt = f"{system_prompt}\n{record['text']}\n答え:\n"
    return {"text": f"{prompt}{record['target']}"}


def evaluate_rouge(
    peft_model,
    base_model_name: str,
    tokenizer,
    raw_eval_dataset,
    system_prompt: str,
    batch_size: int,
    max_new_tokens: int,
    vllm_dtype: str,
    seed: int = 42,
) -> tuple[float, float]:
    if LLM is None or SamplingParams is None:
        raise ImportError("vLLM is required for ROUGE evaluation. Install: pip install vllm")
    if PeftModel is None:
        raise ImportError("peft is required for LoRA merge evaluation. Install: pip install peft")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0. got: {batch_size}")
    if len(raw_eval_dataset) == 0:
        raise ValueError("Validation dataset is empty. ROUGE cannot be computed.")

    custom_tokenizer = JanomeRougeTokenizer(use_stemmer=True)
    scorer = rouge_scorer.RougeScorer(
        ["rougeL"], use_stemmer=False, tokenizer=custom_tokenizer
    )

    prompts: List[str] = []
    references: List[str] = []
    for start in range(0, len(raw_eval_dataset), batch_size):
        batch = raw_eval_dataset[start : start + batch_size]
        if isinstance(batch, dict):
            texts = batch["text"]
            targets = batch["target"]
        else:
            texts = [rec["text"] for rec in batch]
            targets = [rec["target"] for rec in batch]
        prompts.extend([f"{system_prompt}\n{text}\n答え:\n" for text in texts])
        references.extend([target.strip() for target in targets])

    all_scores = []
    with tempfile.TemporaryDirectory(prefix="lora_eval_") as tmp_dir:
        adapter_dir = os.path.join(tmp_dir, "adapter")
        merged_dir = os.path.join(tmp_dir, "merged")

        peft_model.save_pretrained(adapter_dir)

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        merge_model = PeftModel.from_pretrained(base_model, adapter_dir)
        merged_model = merge_model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)

        llm = LLM(model=merged_dir, dtype=vllm_dtype, seed=seed)
        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        outputs = llm.generate(prompts, sampling_params)

        for i, output in enumerate(outputs):
            prediction = output.outputs[0].text.strip()
            score = scorer.score(references[i], prediction)
            all_scores.append(score["rougeL"].fmeasure)

    if not all_scores:
        raise RuntimeError("No ROUGE scores were computed on validation dataset.")

    return float(np.mean(all_scores)), float(np.std(all_scores))


class RougeSFTTrainer(SFTTrainer):
    def __init__(
        self,
        *args,
        rouge_eval_dataset,
        rouge_base_model: str,
        rouge_system_prompt: str,
        rouge_batch_size: int,
        rouge_max_new_tokens: int,
        rouge_vllm_dtype: str,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._rouge_eval_dataset = rouge_eval_dataset
        self._rouge_base_model = rouge_base_model
        self._rouge_system_prompt = rouge_system_prompt
        self._rouge_batch_size = rouge_batch_size
        self._rouge_max_new_tokens = rouge_max_new_tokens
        self._rouge_vllm_dtype = rouge_vllm_dtype

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        if not self.is_world_process_zero():
            return metrics

        rouge_mean, rouge_std = evaluate_rouge(
            self.model,
            self._rouge_base_model,
            self.processing_class if hasattr(self, "processing_class") else self.tokenizer,
            self._rouge_eval_dataset,
            system_prompt=self._rouge_system_prompt,
            batch_size=self._rouge_batch_size,
            max_new_tokens=self._rouge_max_new_tokens,
            vllm_dtype=self._rouge_vllm_dtype,
        )
        metrics[f"{metric_key_prefix}_rougeL_mean"] = rouge_mean
        metrics[f"{metric_key_prefix}_rougeL_std"] = rouge_std
        print(
            f"{metric_key_prefix.upper()} ROUGE-L F1: {rouge_mean:.4f} ± {rouge_std:.4f}"
        )
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen base model を LoRA-SFT するスクリプト")
    parser.add_argument("--train-file", type=str, default=TRAIN_FILE)
    parser.add_argument("--validation-file", type=str, default=VALIDATION_FILE)
    parser.add_argument("--base-model", type=str, default=BASE_MODEL)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument(
        "--system-prompt-file",
        type=str,
        default=SYSTEM_PROMPT_FILE,
        help="system prompt ファイルのパス",
    )
    parser.add_argument("--epochs", type=int, default=1, help="学習エポック数")
    parser.add_argument("--batch-size", type=int, default=8, help="バッチサイズ")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="モデルのdropout率（0.0-1.0未満）",
    )
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="LoRA target modules (comma separated)",
    )
    parser.add_argument(
        "--report-to",
        type=str,
        default="none",
        choices=["none", "wandb", "comet_ml"],
        help="学習ログ送信先",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="qwen-lora-sft",
        help="実験名（WandB/Comet ML に表示）",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="sft-qwen",
        help="WandB/Comet ML のプロジェクト名",
    )
    parser.add_argument(
        "--eval-max-new-tokens",
        type=int,
        default=EVAL_MAX_NEW_TOKENS,
        help="validation ROUGE評価で生成する最大トークン数",
    )
    parser.add_argument(
        "--vllm-dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32", "half"],
        help="vLLM評価時のdtype",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be > 0. got: {args.batch_size}")
    if args.eval_max_new_tokens <= 0:
        raise ValueError(
            f"--eval-max-new-tokens must be > 0. got: {args.eval_max_new_tokens}"
        )

    if LoraConfig is None or get_peft_model is None or TaskType is None:
        raise ImportError(
            "peft is required for LoRA training. Install it with: pip install peft"
        )
    if not (0.0 <= args.dropout < 1.0):
        raise ValueError(f"--dropout must be in [0.0, 1.0). got: {args.dropout}")
    if not (0.0 <= args.lora_dropout < 1.0):
        raise ValueError(
            f"--lora-dropout must be in [0.0, 1.0). got: {args.lora_dropout}"
        )

    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.project_name)
    elif args.report_to == "comet_ml":
        os.environ.setdefault("COMET_PROJECT_NAME", args.project_name)

    with open(args.system_prompt_file, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    ds = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.validation_file},
    )
    train_dataset = ds["train"].map(
        lambda rec: format_record(rec, system_prompt),
        remove_columns=ds["train"].column_names,
    )
    raw_eval_dataset = ds["validation"]
    eval_dataset = raw_eval_dataset.map(
        lambda rec: format_record(rec, system_prompt),
        remove_columns=ds["validation"].column_names,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    for attr_name in (
        "dropout",
        "hidden_dropout",
        "attention_dropout",
        "embd_pdrop",
        "resid_pdrop",
        "summary_first_dropout",
        "classifier_dropout",
    ):
        if hasattr(config, attr_name):
            setattr(config, attr_name, args.dropout)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        dtype=torch.float32,
        trust_remote_code=True,
    )
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = args.dropout

    ##LoRAをかける部分
    target_modules: List[str] = [
        x.strip() for x in args.lora_target_modules.split(",") if x.strip()
    ]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        num_train_epochs=args.epochs,
        learning_rate=2e-5,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[args.report_to] if args.report_to != "none" else [],
        run_name=args.run_name,
        fp16=torch.cuda.is_available(),
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        rouge_eval_dataset=raw_eval_dataset,
        rouge_base_model=args.base_model,
        rouge_system_prompt=system_prompt,
        rouge_batch_size=args.batch_size,
        rouge_max_new_tokens=args.eval_max_new_tokens,
        rouge_vllm_dtype=args.vllm_dtype,
    )
    # Check the base SFTTrainer signature (not RougeSFTTrainer wrapper),
    # because tokenizer/processing_class compatibility depends on TRL version.
    signature_params = inspect.signature(SFTTrainer.__init__).parameters
    if "dataset_text_field" in signature_params:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in signature_params:
        trainer_kwargs["max_seq_length"] = 1024
    if "processing_class" in signature_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    if "callbacks" in signature_params:
        trainer_kwargs["callbacks"] = [EarlyStoppingCallback(early_stopping_patience=2)]

    trainer = RougeSFTTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
