import argparse
import os

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="XLSum Japanese dataset を CSV で保存する"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="csebuetnlp/xlsum",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="japanese",
        help="dataset config name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="xlsum_ja",
        help="CSV 出力先ディレクトリ",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_dataset(args.dataset, args.config)
    for split in dataset.keys():
        out_path = os.path.join(args.output_dir, f"{split}.csv")
        dataset[split].to_csv(out_path, index=False)
        print(f"Saved: {out_path} ({len(dataset[split])} rows)")

    print("Done.")


if __name__ == "__main__":
    main()
