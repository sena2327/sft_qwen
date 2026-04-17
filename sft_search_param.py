import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def format_lr(lr: float) -> str:
    return f"{lr:.0e}".replace("+0", "").replace("+", "")


def run_cmd(cmd: list[str], env: dict[str, str]) -> None:
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def wait_for_model_dir(model_dir: Path, timeout_sec: int = 300) -> Path:
    deadline = time.time() + timeout_sec
    last_log_at = 0.0

    def pick_ready_dir() -> Path | None:
        root_config = model_dir / "config.json"
        if model_dir.is_dir() and root_config.exists():
            return model_dir

        # save_model が未実行でも checkpoint-* に config がある場合はそれを評価に使う
        if model_dir.is_dir():
            checkpoints = sorted(
                model_dir.glob("checkpoint-*"),
                key=lambda p: p.name,
            )
            for ckpt in reversed(checkpoints):
                if (ckpt / "config.json").exists():
                    return ckpt
        return None

    while time.time() < deadline:
        ready_dir = pick_ready_dir()
        if ready_dir is not None:
            return ready_dir
        now = time.time()
        if now - last_log_at >= 5.0:
            print(f"waiting for model artifacts... dir={model_dir}")
            last_log_at = now
        time.sleep(1.0)

    root_exists = model_dir.is_dir()
    has_root_config = (model_dir / "config.json").exists()
    checkpoints = sorted(model_dir.glob("checkpoint-*")) if root_exists else []
    raise FileNotFoundError(
        "model artifacts are not ready for evaluation. "
        f"dir={model_dir}, root_exists={root_exists}, "
        f"has_root_config={has_root_config}, "
        f"checkpoints={[p.name for p in checkpoints]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SFTのaccumulation/lr探索を外部実行で行う"
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default=".",
        help="sft.py と generate_evaluate.py がある作業ディレクトリ",
    )
    parser.add_argument(
        "--keep-model-on-fail",
        action="store_true",
        help="失敗時に学習済みモデルディレクトリを削除しない",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default="0",
        help="sft.py / generate_evaluate.py 実行時の CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument(
        "--model-ready-timeout-sec",
        type=int,
        default=300,
        help="学習後に評価用モデルが揃うまで待つ秒数",
    )
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    sft_script = workdir / "sft.py"
    eval_script = workdir / "generate_evaluate.py"
    if not sft_script.exists():
        raise FileNotFoundError(f"not found: {sft_script}")
    if not eval_script.exists():
        raise FileNotFoundError(f"not found: {eval_script}")
    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    child_env["PYTHONUNBUFFERED"] = "1"
    print(f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}")

    accum_list = [1, 8, 16, 32]
    lr_list = [1e-3, 1e-4, 1e-5, 1e-6]
    total = len(accum_list) * len(lr_list)
    trial = 0

    for accum in accum_list:
        for lr in lr_list:
            trial += 1
            lr_tag = format_lr(lr)
            model_dir = workdir / f"sft_param_1_{lr_tag}_acc{accum}"

            print("=" * 72)
            print(f"[{trial}/{total}] accum={accum}, lr={lr:g}")

            sft_cmd = [
                sys.executable,
                str(sft_script),
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--gradient-accumulation-steps",
                str(accum),
                "--learning-rate",
                str(lr),
                "--output-dir",
                str(model_dir),
            ]

            succeeded = False
            try:
                run_cmd(sft_cmd, env=child_env)
                print(f"sft finished. checking model dir: {model_dir}")
                eval_model_dir = wait_for_model_dir(
                    model_dir, timeout_sec=args.model_ready_timeout_sec
                )
                print(f"model ready for eval: {eval_model_dir}")
                eval_cmd = [
                    sys.executable,
                    str(eval_script),
                    "--model",
                    str(eval_model_dir),
                ]
                run_cmd(eval_cmd, env=child_env)
                succeeded = True
            finally:
                if model_dir.exists() and (succeeded or not args.keep_model_on_fail):
                    print(f"remove model dir: {model_dir}")
                    shutil.rmtree(model_dir, ignore_errors=True)

    print("all parameter trials finished.")


if __name__ == "__main__":
    main()
