"""
掃過 results/ 底下所有跑完的實驗 (有 results.csv 的資料夾)，彙整成一張總表，
方便跨實驗比較 (不同 k、dsc_hidden、merged vs per-group...)。

用法:
  python compare_experiments.py
  python compare_experiments.py --root results
"""

import argparse
import csv
from pathlib import Path


def read_mean_row(results_csv: Path):
    with open(results_csv) as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row["Row Names"] == "Mean":
            return row
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    exps = []
    for d in sorted(args.root.iterdir()):
        csv_path = d / "results.csv"
        if not csv_path.exists():
            continue
        mean = read_mean_row(csv_path)
        if mean is None:
            continue
        with open(csv_path) as f:
            n_rows = sum(1 for _ in csv.DictReader(f)) - 1  # 扣掉 Mean 那列
        exps.append((d.name, n_rows, mean))

    if not exps:
        raise SystemExit(f"{args.root} 底下找不到任何跑完的 results.csv")

    lines = []
    lines.append(f"Cross-experiment comparison — {args.root}")
    lines.append(f"共 {len(exps)} 個實驗")
    lines.append("=" * 100)
    lines.append("")
    header = f"{'experiment':<32} {'#class':>6} | {'I-AUROC':>8} {'P-AUROC':>8} {'P-PRO':>7} {'AP':>7}"
    has_f1 = any("best_f1" in m for _, _, m in exps)
    if has_f1:
        header += f" | {'best_f1':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    for name, n_rows, mean in sorted(exps, key=lambda x: -float(x[2].get("image_auroc", 0) or 0)):
        row = (f"{name:<32} {n_rows:>6} | "
              f"{float(mean.get('image_auroc', 0) or 0):>8.4f} "
              f"{float(mean.get('pixel_auroc', 0) or 0):>8.4f} "
              f"{float(mean.get('pixel_pro', 0) or 0):>7.4f} "
              f"{float(mean.get('image_ap', 0) or 0):>7.4f}")
        if has_f1:
            v = mean.get("best_f1", "")
            row += f" | {float(v) * 100:>7.2f}%" if v else f" | {'N/A':>8}"
        lines.append(row)

    lines.append("")
    lines.append("=" * 100)
    lines.append("排序依 I-AUROC (mean，跨該實驗所有 class/group 平均) 由高到低")
    lines.append("#class = 該實驗訓練了幾個獨立 class/group")
    lines.append("每個實驗的完整明細 (張數、混淆矩陣) 看各自的 analyze results/report.txt")
    lines.append("(用 summarize_results.py 針對該資料夾產生)")

    out_path = args.out or (args.root / "compare_experiments.txt")
    out_path.write_text("\n".join(lines))
    print(f"寫入 {out_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
