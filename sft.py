import argparse
import inspect
import os
from typing import Dict

import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
TRAIN_FILE = "data/train.jsonl"
VALIDATION_FILE = "data/validation.jsonl"
OUTPUT_DIR = "sft_output"
SYSTEM_PROMPT_FILE = "system_prompt.txt"


def format_record(record: Dict[str, str], system_prompt: str) -> Dict[str, str]:
    prompt = f"{system_prompt}\n{record['text']}\n答え:\n"
    return {"text": f"{prompt}{record['target']}"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen base model を SFT するスクリプト")
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
    parser.add_argument("--epochs", type=int,default=1, help="学習エポック数")
    parser.add_argument("--batch-size", type=int, default=8, help="バッチサイズ")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="モデルのdropout率（0.0-1.0未満）",
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
        default="qwen-sft",
        help="実験名（WandB/Comet ML に表示）",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="sft-qwen",
        help="WandB/Comet ML のプロジェクト名",
    )
    args = parser.parse_args()
    if not (0.0 <= args.dropout < 1.0):
        raise ValueError(f"--dropout must be in [0.0, 1.0). got: {args.dropout}")

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
    eval_dataset = ds["validation"].map(
        lambda rec: format_record(rec, system_prompt),
        remove_columns=ds["validation"].column_names,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = args.dropout

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
        report_to=[args.report_to] if args.report_to != "none" else [],
        run_name=args.run_name,
        fp16=torch.cuda.is_available(),
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=1024,
    )
    signature_params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in signature_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
