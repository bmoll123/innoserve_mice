"""
把一次 --results_path 底下所有 group 的結果彙整成一份 report.txt。

資料來源 (只讀現有檔案，不會重新推論):
  <results_path>/results.csv                                   每 group 一列: AUROC/AP/PRO/best_epoch(/best_f1，若有)
  <results_path>/analyze results/report_mvtec_<gid>.txt         固定門檻下的混淆矩陣/acc/recall
  <results_path>/analyze results/training_log_mvtec_<gid>.csv   最後一列: oracle 門檻 (best_acc/best_balanced_acc/best_f1)
  <data_path>/<gid>/{train/good, test/good, test/defect, other_fake}  各資料夾張數

用法:
  python summarize_results.py --results_path results/pcb_groups_k=0.25
"""

import argparse
import csv
import re
from pathlib import Path


def count_images(d: Path):
    """回傳張數；資料夾不存在回傳 None (跟「存在但是空的」=0 區分開)。"""
    if not d.is_dir():
        return None
    return sum(1 for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def find_anomaly_dir(group_dir: Path):
    """不同資料集的異常資料夾命名不一樣: PCB 用 defect，鞋子用 fake。"""
    for name in ("defect", "fake"):
        d = group_dir / "test" / name
        if d.is_dir():
            return d
    return group_dir / "test" / "defect"  # 都沒有時回傳預設路徑 (count_images 會回 None)


def fmt_count(v):
    return "N/A" if v is None else str(v)


def read_results_csv(path: Path):
    rows = {}
    if not path.exists():
        return rows
    with open(path) as f:
        for row in csv.DictReader(f):
            name = row["Row Names"]
            if name == "Mean":
                continue
            gid = name.replace("mvtec_", "")
            rows[gid] = row
    return rows


def read_training_log_last_row(path: Path):
    if not path.exists():
        return {}
    with open(path) as f:
        rows = list(csv.DictReader(f))
    # 最後一列不一定是有評估的那列 (eval_epochs 間隔)，往回找第一列有算過分類指標的
    for row in reversed(rows):
        if row.get("best_threshold", "0") not in ("", "0", "0.0000"):
            return row
    return rows[-1] if rows else {}


def parse_report_txt(path: Path):
    if not path.exists():
        return {}
    text = path.read_text()
    out = {}
    patterns = {
        "threshold": r"^threshold\s*:\s*([\d.]+)",
        "accuracy": r"^accuracy\s*:\s*([\d.]+)",
        "balanced_accuracy": r"^balanced accuracy\s*:\s*([\d.]+)",
        "f1": r"^f1 \(fake\)\s*:\s*([\d.]+)",
        "recall_fake": r"^recall\s+fake / good\s*:\s*([\d.]+)",
        "recall_good": r"^recall\s+fake / good\s*:\s*[\d.]+\s*%\s*/\s*([\d.]+)",
        "wrong": r"^wrong predictions\s*:\s*(\d+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, re.MULTILINE)
        if m:
            out[key] = m.group(1)
    cm = re.search(r"good\s+(\d+)\s+(\d+)\n\s*fake\s+(\d+)\s+(\d+)", text)
    if cm:
        out["tn"], out["fp"], out["fn"], out["tp"] = cm.groups()
    return out


def fnum(v, default="N/A"):
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return default


def fpct(v, default="N/A"):
    try:
        x = float(v)
        x = x * 100 if x <= 1.0 else x
        return f"{x:.2f}%"
    except (TypeError, ValueError):
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_path", type=Path, required=True)
    ap.add_argument("--data_path", type=Path, default=Path("/home/yuyun/Desktop/Innoserve/deeppcb_mice"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    results_csv = args.results_path / "results.csv"
    analyze_dir = args.results_path / "analyze results"
    out_path = args.out or (analyze_dir / "report.txt")

    csv_rows = read_results_csv(results_csv)
    if not csv_rows:
        raise SystemExit(f"讀不到 {results_csv}，確認 --results_path 對不對")

    gids = sorted(csv_rows.keys())
    data_missing = not args.data_path.is_dir()
    lines = []
    lines.append(f"Summary report — {args.results_path}")
    lines.append(f"共 {len(gids)} 個 group")
    if data_missing:
        lines.append(f"注意: 資料集路徑 {args.data_path} 已不存在，張數欄位全部顯示 N/A")
    lines.append("=" * 100)
    lines.append("")

    # ── 總覽表 ──────────────────────────────────────────────
    header = (f"{'group':<8} {'train':>6} {'test_g':>7} {'test_f':>7} {'other':>6} | "
              f"{'I-AUROC':>8} {'P-AUROC':>8} {'P-PRO':>7} {'epoch':>6} | "
              f"{'acc':>7} {'bacc':>7} {'thr':>6} | {'best_bacc':>9} {'best_f1':>8}")
    lines.append(header)
    lines.append("-" * len(header))

    agg = {"train": 0, "test_g": 0, "test_f": 0, "other": 0}
    auroc_vals, bacc_vals = [], []

    for gid in gids:
        g = args.data_path / gid
        n_train = count_images(g / "train" / "good")
        n_test_g = count_images(g / "test" / "good")
        n_test_f = count_images(find_anomaly_dir(g))
        n_other = count_images(g / "other_fake")
        for key, v in (("train", n_train), ("test_g", n_test_g), ("test_f", n_test_f), ("other", n_other)):
            agg[key] += v or 0

        row = csv_rows[gid]
        report = parse_report_txt(analyze_dir / f"report_mvtec_{gid}.txt")
        last = read_training_log_last_row(analyze_dir / f"training_log_mvtec_{gid}.csv")

        i_auroc = float(row.get("image_auroc", 0) or 0)
        p_auroc = float(row.get("pixel_auroc", 0) or 0)
        p_pro = float(row.get("pixel_pro", 0) or 0)
        epoch = row.get("best_epoch", "N/A")
        best_bacc = last.get("best_balanced_acc", "")
        best_f1_col = row.get("best_f1") or last.get("best_f1", "")

        if best_bacc:
            bacc_vals.append(float(best_bacc))
        auroc_vals.append(i_auroc)

        lines.append(
            f"{gid:<8} {fmt_count(n_train):>6} {fmt_count(n_test_g):>7} "
            f"{fmt_count(n_test_f):>7} {fmt_count(n_other):>6} | "
            f"{i_auroc:>8.4f} {p_auroc:>8.4f} {p_pro:>7.4f} {epoch:>6} | "
            f"{fpct(report.get('accuracy'), '  N/A'):>7} "
            f"{fpct(report.get('balanced_accuracy'), '  N/A'):>7} "
            f"{fnum(report.get('threshold'), ' N/A'):>6} | "
            f"{fpct(best_bacc, '     N/A'):>9} "
            f"{fpct(best_f1_col, '    N/A'):>8}"
        )

    lines.append("-" * len(header))
    n = len(gids)
    lines.append(
        f"{'TOTAL':<8} {agg['train']:>6} {agg['test_g']:>7} {agg['test_f']:>7} {agg['other']:>6} | "
        f"{'mean':>8} {'':>8} {'':>7} {'':>6} |"
    )
    lines.append(f"  mean I-AUROC      : {sum(auroc_vals)/n:.4f}")
    if bacc_vals:
        lines.append(f"  mean best_bacc     : {sum(bacc_vals)/len(bacc_vals)*100:.2f}%  "
                     f"(oracle 上界，用 test 標籤挑出來的門檻，不是可部署數字)")
    lines.append("")
    lines.append("=" * 100)
    lines.append("欄位說明")
    lines.append("=" * 100)
    lines.append("train/test_g/test_f/other : train/good, test/good, test/defect, other_fake 各自張數")
    lines.append("I-AUROC/P-AUROC/P-PRO     : 門檻無關的排序/分割指標，來自 results.csv")
    lines.append("epoch                     : 被選為 best ckpt 的 epoch")
    lines.append("acc/bacc/thr              : 目前設定的固定門檻下的表現，來自 report_mvtec_<gid>.txt")
    lines.append("best_bacc/best_f1         : oracle 門檻 (掃描 test 分數找出來的上界)，不代表可部署的真實表現")
    lines.append("")

    # ── 每個 group 的細節 (混淆矩陣) ──────────────────────────
    lines.append("=" * 100)
    lines.append("各 group 混淆矩陣 (固定門檻)")
    lines.append("=" * 100)
    for gid in gids:
        report = parse_report_txt(analyze_dir / f"report_mvtec_{gid}.txt")
        if not report:
            continue
        lines.append(f"\n[{gid}]  threshold={report.get('threshold', 'N/A')}  "
                     f"wrong={report.get('wrong', 'N/A')}")
        if "tp" in report:
            lines.append(f"          predicted good  predicted fake")
            lines.append(f"  good    {report['tn']:>12}  {report['fp']:>14}")
            lines.append(f"  fake    {report['fn']:>12}  {report['tp']:>14}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"寫入 {out_path}")
    print(f"共彙整 {n} 個 group，總計 train={agg['train']} test_good={agg['test_g']} "
          f"test_defect={agg['test_f']} other_fake={agg['other']}")


if __name__ == "__main__":
    main()
