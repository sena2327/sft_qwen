import argparse
import json
import os

from datasets import load_dataset


def save_jsonl(split_ds, out_path: str) -> int:
    rows = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in split_ds:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="mkshing/xlsum_ja を JSONL で保存する"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mkshing/xlsum_ja",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="xlsum_ja",
        help="JSONL 出力先ディレクトリ",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_dataset(args.dataset)
    for split in dataset.keys():
        out_path = os.path.join(args.output_dir, f"{split}.jsonl")
        n = save_jsonl(dataset[split], out_path)
        print(f"Saved: {out_path} ({n} rows)")

    print("Done.")


if __name__ == "__main__":
    main()
