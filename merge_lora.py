import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LoRA adapter をベースモデルにマージして保存する"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-0.6B-Base",
        help="ベースモデルのHF IDまたはローカルパス",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default="sft_output_lora",
        help="LoRA adapter の保存ディレクトリ",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="sft_output_lora_merged",
        help="マージ済みモデルの保存先",
    )
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter: {args.lora_path}")
    peft_model = PeftModel.from_pretrained(base_model, args.lora_path)

    print("Merging adapter into base model...")
    merged_model = peft_model.merge_and_unload()

    print(f"Saving merged model to: {args.output_dir}")
    merged_model.save_pretrained(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
