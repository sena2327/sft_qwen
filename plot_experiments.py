import argparse
import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt


DEFAULT_EXPERIMENT_TIMESTAMPS = {
    "Exp1": "20260416_153921",
    "Exp2": "20260414_143658",
    "Exp3": "20260414_183125",
    "Exp4": "20260414_202420",
    "Exp5": "20260415_172138",
    "Exp6": "20260415_215149",
}

EXPERIMENT_DISPLAY_NAMES = {
    "Exp1": "prompt tuning",
    "Exp2": "naive sft_fft",
    "Exp3": "lora",
    "Exp4": "sft+lora",
    "Exp5": "summary",
    "Exp6": "sft+dpo",
}


def load_rows(csv_path: str) -> Dict[str, Dict[str, str]]:
    rows_by_ts: Dict[str, Dict[str, str]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_by_ts[row["Timestamp"]] = row
    return rows_by_ts


def parse_mapping(items: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --map format: {item}. Expected ExpX=YYYYMMDD_HHMMSS")
        k, v = item.split("=", 1)
        mapping[k.strip()] = v.strip()
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description="output.csv から実験1〜6のROUGEグラフを作成する"
    )
    parser.add_argument("--csv", type=str, default="output.csv", help="入力CSV")
    parser.add_argument(
        "--output",
        type=str,
        default="experiments_rouge.png",
        help="出力画像ファイル(PNG)",
    )
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        help="実験とタイムスタンプの対応を上書き (例: Exp3=20260414_183125)",
    )
    args = parser.parse_args()

    rows = load_rows(args.csv)
    ts_map = dict(DEFAULT_EXPERIMENT_TIMESTAMPS)
    ts_map.update(parse_mapping(args.map))

    labels: List[str] = []
    means: List[float] = []
    stds: List[float] = []
    meta: List[str] = []

    for exp_name in sorted(ts_map.keys(), key=lambda x: int(x.replace("Exp", ""))):
        ts = ts_map[exp_name]
        if ts not in rows:
            raise KeyError(f"Timestamp not found in {args.csv}: {ts} ({exp_name})")
        row = rows[ts]
        labels.append(f"{exp_name}: {EXPERIMENT_DISPLAY_NAMES.get(exp_name, exp_name)}")
        means.append(float(row["ROUGE-L_Mean"]))
        stds.append(float(row["ROUGE-L_Std"]))
        meta.append(row["Model_Path"])

    plt.figure(figsize=(12, 6))
    x = range(len(labels))
    bars = plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(list(x), labels, rotation=20, ha="right")
    plt.ylabel("ROUGE-L F1")
    plt.title("Experiment Results (Mean ± Std)")
    plt.ylim(0, max(m + s for m, s in zip(means, stds)) * 1.15)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for i, (bar, m, s, model) in enumerate(zip(bars, means, stds, meta)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{m:.4f}\n±{s:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        plt.text(
            i,
            0.002,
            model,
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
            alpha=0.7,
        )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=160)
    print(f"Saved graph: {args.output}")


if __name__ == "__main__":
    main()
