import argparse
import csv
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_RE = re.compile(r"sft_param_(?P<batch>\d+)_(?P<lr>1e-[0-9]+)_acc(?P<acc>\d+)$")


def parse_model_path(model_path: str) -> tuple[int, str, int] | None:
    name = Path(model_path).name
    m = MODEL_RE.match(name)
    if not m:
        return None
    return int(m.group("batch")), m.group("lr"), int(m.group("acc"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="output.csv から sft_param 実験のヒートマップを描画する"
    )
    parser.add_argument("--csv", type=str, default="output.csv", help="入力CSV")
    parser.add_argument(
        "--output",
        type=str,
        default="sft_param_heatmap.png",
        help="出力画像ファイル(PNG)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="表示対象の batch_size",
    )
    args = parser.parse_args()

    # key: (acc, lr_str) -> latest row by timestamp
    latest_by_key: dict[tuple[int, str], dict[str, str]] = {}
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = parse_model_path(row["Model_Path"])
            if parsed is None:
                continue
            batch_size, lr_str, acc = parsed
            if batch_size != args.batch_size:
                continue
            key = (acc, lr_str)
            if key not in latest_by_key or row["Timestamp"] > latest_by_key[key]["Timestamp"]:
                latest_by_key[key] = row

    if not latest_by_key:
        raise ValueError(
            f"sft_param data not found in {args.csv} for batch_size={args.batch_size}"
        )

    acc_values = sorted({acc for acc, _ in latest_by_key.keys()})
    lr_values = sorted({lr for _, lr in latest_by_key.keys()}, key=lambda x: float(x))

    mean_mat = np.full((len(acc_values), len(lr_values)), np.nan, dtype=float)
    std_mat = np.full((len(acc_values), len(lr_values)), np.nan, dtype=float)

    acc_to_i = {acc: i for i, acc in enumerate(acc_values)}
    lr_to_j = {lr: j for j, lr in enumerate(lr_values)}

    for (acc, lr), row in latest_by_key.items():
        i = acc_to_i[acc]
        j = lr_to_j[lr]
        mean_mat[i, j] = float(row["ROUGE-L_Mean"])
        std_mat[i, j] = float(row["ROUGE-L_Std"])

    valid_means = mean_mat[~np.isnan(mean_mat)]
    vmin = float(valid_means.min()) if valid_means.size > 0 else 0.0
    vmax = float(valid_means.max()) if valid_means.size > 0 else 1.0

    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(mean_mat, cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("ROUGE-L Mean")

    ax.set_xticks(range(len(lr_values)))
    ax.set_yticks(range(len(acc_values)))
    ax.set_xticklabels(lr_values)
    ax.set_yticklabels([str(a) for a in acc_values])
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Gradient Accumulation Steps")
    ax.set_title(f"SFT Param Search Heatmap (batch_size={args.batch_size})")

    for i in range(len(acc_values)):
        for j in range(len(lr_values)):
            if np.isnan(mean_mat[i, j]):
                text = "N/A"
                color = "black"
            else:
                text = f"{mean_mat[i, j]:.4f}\n±{std_mat[i, j]:.4f}"
                threshold = (vmin + vmax) / 2.0
                color = "white" if mean_mat[i, j] >= threshold else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=180)
    print(f"Saved heatmap: {args.output}")


if __name__ == "__main__":
    main()
