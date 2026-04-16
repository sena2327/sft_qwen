import argparse
import itertools
import json
import os
from typing import Dict, List, Tuple

try:
    from scipy import stats
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "scipy is required for t-test. Install with: pip install scipy\n"
        f"import error: {e}"
    )


DEFAULT_RUNS: List[Tuple[str, str]] = [
    ("Exp1", "20260414_103915"),
    ("Exp2", "20260414_143658"),
    ("Exp3", "20260414_183125"),
    ("Exp4", "20260414_202420"),
    ("Exp5", "20260415_172138"),
    ("Exp6", "20260415_215149"),
]


def load_rouge_scores(path: str) -> List[float]:
    scores: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "rouge_L_f1" not in obj:
                raise KeyError(f"'rouge_L_f1' not found in {path}")
            scores.append(float(obj["rouge_L_f1"]))
    if not scores:
        raise ValueError(f"No scores loaded from {path}")
    return scores


def parse_runs(values: List[str]) -> List[Tuple[str, str]]:
    if not values:
        return DEFAULT_RUNS
    parsed: List[Tuple[str, str]] = []
    for v in values:
        if "=" not in v:
            raise ValueError(f"Invalid --run format: {v}. Use LABEL=TIMESTAMP")
        label, ts = v.split("=", 1)
        parsed.append((label.strip(), ts.strip()))
    if len(parsed) != 6:
        raise ValueError(f"Exactly 6 runs are required, got {len(parsed)}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="logs/*_details.jsonl の rouge_L_f1 で6実験15ペアのt検定を行う"
    )
    parser.add_argument("--logs-dir", type=str, default="logs")
    parser.add_argument(
        "--output",
        type=str,
        default="logs/ttest_6runs_15pairs.txt",
        help="結果テキスト出力先",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="実験指定 (例: --run Exp1=20260414_103915)。6個指定。",
    )
    args = parser.parse_args()

    runs = parse_runs(args.run)
    series: Dict[str, List[float]] = {}
    timestamps: Dict[str, str] = {}

    for label, ts in runs:
        path = os.path.join(args.logs_dir, f"{ts}_details.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing log file: {path}")
        series[label] = load_rouge_scores(path)
        timestamps[label] = ts

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    lines: List[str] = []
    lines.append("T-test results for 6 runs (15 pairs)")
    lines.append("=" * 80)
    lines.append("Method: paired t-test when sample counts are equal; Welch otherwise")
    lines.append("")
    lines.append("Runs:")
    for label, ts in runs:
        vals = series[label]
        lines.append(
            f"- {label}: ts={ts}, n={len(vals)}, mean={sum(vals)/len(vals):.6f}, std={stats.tstd(vals):.6f}"
        )
    lines.append("")
    lines.append("Pairwise tests:")

    pair_idx = 0
    for (label_a, _), (label_b, _) in itertools.combinations(runs, 2):
        pair_idx += 1
        a = series[label_a]
        b = series[label_b]
        n_a = len(a)
        n_b = len(b)
        if n_a == n_b:
            test_name = "paired_ttest"
            res = stats.ttest_rel(a, b, alternative="two-sided")
            df = n_a - 1
        else:
            test_name = "welch_ttest"
            res = stats.ttest_ind(a, b, equal_var=False, alternative="two-sided")
            # Welch-Satterthwaite df
            var_a = stats.tvar(a)
            var_b = stats.tvar(b)
            df = (var_a / n_a + var_b / n_b) ** 2 / (
                (var_a**2) / ((n_a**2) * (n_a - 1)) + (var_b**2) / ((n_b**2) * (n_b - 1))
            )

        lines.append(
            f"[{pair_idx:02d}/15] {label_a} vs {label_b} | {test_name} | "
            f"t={res.statistic:.6f}, p={res.pvalue:.6e}, df={df:.2f}"
        )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
