import argparse
import glob
import os
import subprocess
import sys


def current_detail_logs(log_dir: str = "logs") -> set[str]:
    pattern = os.path.join(log_dir, "*_details.jsonl")
    return set(glob.glob(pattern))


def main():
    parser = argparse.ArgumentParser(
        description="system_prompts 配下の prompt を順に使って generate_evaluate.py を実行する"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sft_output",
        help="generate_evaluate.py に渡すモデルパス",
    )
    parser.add_argument(
        "--prompts-dir",
        type=str,
        default="system_prompts",
        help="system prompt ファイル群があるディレクトリ",
    )
    args = parser.parse_args()

    pattern = os.path.join(args.prompts_dir, "system_prompt_*.txt")
    prompt_files = sorted(glob.glob(pattern))

    if not prompt_files:
        print(f"❌ Prompt files not found: {pattern}")
        sys.exit(1)

    print(f"Found {len(prompt_files)} prompt files.")

    for idx, prompt_file in enumerate(prompt_files, start=1):
        print("=" * 60)
        print(f"[{idx}/{len(prompt_files)}] Running with: {prompt_file}")
        cmd = [
            sys.executable,
            "generate_evaluate.py",
            "--model",
            args.model,
            "--system-prompt-file",
            prompt_file,
        ]
        print("$ " + " ".join(cmd))
        before_logs = current_detail_logs()
        subprocess.run(cmd, check=True)
        after_logs = current_detail_logs()
        new_logs = sorted(after_logs - before_logs)
        for log_path in new_logs:
            os.remove(log_path)
            print(f"🧹 Deleted generated detail log: {log_path}")

    print("=" * 60)
    print("✅ All prompt evaluations finished.")


if __name__ == "__main__":
    main()
