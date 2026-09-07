"""
把一次 --results_path 底下所有 group 的最終測試結果彙整成一份 report.txt。

資料來源 (只讀現有檔案，不會重新推論):
  <results_path>/results.csv                                 每 group 一列: best_epoch (訓練資訊，非分類指標)
  <results_path>/analyze results/<gid>/report.txt             final_test() 產生，展開 test/good vs
                                                                (test/defect+other_fake) 的完整測試結果
  <data_path>/<gid>/{train/good, test/good, test/defect, other_fake}  各資料夾張數

report.txt 裡的 accuracy/balanced accuracy/precision/recall/f1/I-AUROC/P-AUROC/P-PRO
全部來自同一次評估 (final_test，門檻沿用驗證集校準出來的 self.threshold，不是用這批
test 的答案重新挑門檻)，是這個 group 唯一一份「正式」的測試報告。

用法:
  python summarize_results.py --results_path "results/pcb_groups_k=0.25"
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


def parse_report_txt(path: Path):
    if not path.exists():
        return {}
    text = path.read_text()
    out = {}
    patterns = {
        "threshold": r"^threshold\s*:\s*([\d.]+)",
        "pixel_threshold": r"^pixel threshold\s*:\s*([\d.]+)",
        "accuracy": r"^accuracy\s*:\s*([\d.]+)",
        "balanced_accuracy": r"^balanced accuracy\s*:\s*([\d.]+)",
        "precision": r"^precision \(fake\)\s*:\s*([\d.]+)",
        "f1": r"^f1 \(fake\)\s*:\s*([\d.]+)",
        "recall_fake": r"^recall\s+fake / good\s*:\s*([\d.]+)",
        "recall_good": r"^recall\s+fake / good\s*:\s*[\d.]+\s*%\s*/\s*([\d.]+)",
        "i_auroc": r"^I-AUROC\s*:\s*([\d.]+)",
        "p_auroc": r"^P-AUROC\s*:\s*([\d.]+)",
        "p_pro": r"^P-PRO\s*:\s*([\d.]+)",
        "ap_bbox": r"^AP@0\.5\(bbox\)\s*:\s*([\d.]+)",
        "miss_rate": r"^Miss Rate\s*:\s*([\d.]+)",
        "false_alarm": r"^False Alarm\s*:\s*([\d.]+)",
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

    if not analyze_dir.is_dir():
        raise SystemExit(f"讀不到 {analyze_dir}，確認 --results_path 對不對，且已經跑過 final_test()")

    gids = sorted(p.name for p in analyze_dir.iterdir()
                 if p.is_dir() and (p / "report.txt").exists())
    if not gids:
        raise SystemExit(f"{analyze_dir} 底下沒有任何 <group_id>/report.txt，"
                         f"確認 main.py 有跑完 tester() (需要有 ckpt_best 才會執行 final_test)")

    csv_rows = read_results_csv(results_csv)
    data_missing = not args.data_path.is_dir()
    lines = []
    lines.append(f"Summary report — {args.results_path}")
    lines.append(f"共 {len(gids)} 個 group  (final_test: test/good vs test/defect+other_fake)")
    if data_missing:
        lines.append(f"注意: 資料集路徑 {args.data_path} 已不存在，張數欄位全部顯示 N/A")
    lines.append("=" * 112)
    lines.append("")

    # ── 總覽表 ──────────────────────────────────────────────
    header = (f"{'group':<8} {'train':>6} {'test_g':>7} {'test_f':>7} {'other':>6} {'epoch':>6} | "
              f"{'cls_thr':>7} {'pix_thr':>7} | "
              f"{'acc':>7} {'bacc':>7} {'prec':>7} {'rec_f':>7} {'rec_g':>7} {'f1':>7} | "
              f"{'I-AUROC':>8} {'P-AUROC':>8} {'P-PRO':>7} {'AP@.5':>7} {'Miss':>7} {'FalseAl':>8}")
    lines.append(header)
    lines.append("-" * len(header))

    agg = {"train": 0, "test_g": 0, "test_f": 0, "other": 0}
    reports = {}
    auroc_vals, bacc_vals, f1_vals, ap_vals, miss_vals, fa_vals = [], [], [], [], [], []

    for gid in gids:
        g = args.data_path / gid
        n_train = count_images(g / "train" / "good")
        n_test_g = count_images(g / "test" / "good")
        n_test_f = count_images(find_anomaly_dir(g))
        n_other = count_images(g / "other_fake")
        for key, v in (("train", n_train), ("test_g", n_test_g), ("test_f", n_test_f), ("other", n_other)):
            agg[key] += v or 0

        report = parse_report_txt(analyze_dir / gid / "report.txt")
        reports[gid] = report
        epoch = (csv_rows.get(gid, {}) or {}).get("best_epoch", "N/A")

        i_auroc = float(report.get("i_auroc", 0) or 0) / 100
        p_auroc = float(report.get("p_auroc", 0) or 0) / 100
        p_pro = float(report.get("p_pro", 0) or 0) / 100
        bacc = report.get("balanced_accuracy")
        f1 = report.get("f1")

        if report.get("i_auroc"):
            auroc_vals.append(i_auroc)
        if bacc:
            bacc_vals.append(float(bacc) / 100)
        if f1:
            f1_vals.append(float(f1) / 100)
        if report.get("ap_bbox"):
            ap_vals.append(float(report["ap_bbox"]) / 100)
        if report.get("miss_rate"):
            miss_vals.append(float(report["miss_rate"]) / 100)
        if report.get("false_alarm"):
            fa_vals.append(float(report["false_alarm"]) / 100)

        lines.append(
            f"{gid:<8} {fmt_count(n_train):>6} {fmt_count(n_test_g):>7} "
            f"{fmt_count(n_test_f):>7} {fmt_count(n_other):>6} {epoch:>6} | "
            f"{fnum(report.get('threshold'), '    N/A'):>7} "
            f"{fnum(report.get('pixel_threshold'), '    N/A'):>7} | "
            f"{fpct(report.get('accuracy'), '   N/A'):>7} "
            f"{fpct(bacc, '   N/A'):>7} "
            f"{fpct(report.get('precision'), '   N/A'):>7} "
            f"{fpct(report.get('recall_fake'), '   N/A'):>7} "
            f"{fpct(report.get('recall_good'), '   N/A'):>7} "
            f"{fpct(f1, '   N/A'):>7} | "
            f"{fpct(report.get('i_auroc'), '   N/A'):>8} "
            f"{fpct(report.get('p_auroc'), '   N/A'):>8} "
            f"{fpct(report.get('p_pro'), '   N/A'):>7} "
            f"{fpct(report.get('ap_bbox'), '   N/A'):>7} "
            f"{fpct(report.get('miss_rate'), '   N/A'):>7} "
            f"{fpct(report.get('false_alarm'), '   N/A'):>8}"
        )

    lines.append("-" * len(header))
    n = len(gids)
    lines.append(f"{'TOTAL':<8} {agg['train']:>6} {agg['test_g']:>7} {agg['test_f']:>7} {agg['other']:>6}")
    if auroc_vals:
        lines.append(f"  mean I-AUROC          : {sum(auroc_vals)/len(auroc_vals)*100:.2f}%")
    if bacc_vals:
        lines.append(f"  mean balanced accuracy : {sum(bacc_vals)/len(bacc_vals)*100:.2f}%")
    if f1_vals:
        lines.append(f"  mean f1 (fake)         : {sum(f1_vals)/len(f1_vals)*100:.2f}%")
    if ap_vals:
        lines.append(f"  mean AP@0.5 (bbox)     : {sum(ap_vals)/len(ap_vals)*100:.2f}%")
    if miss_vals:
        lines.append(f"  mean Miss Rate         : {sum(miss_vals)/len(miss_vals)*100:.2f}%")
    if fa_vals:
        lines.append(f"  mean False Alarm       : {sum(fa_vals)/len(fa_vals)*100:.2f}%")
    lines.append("")
    lines.append("=" * 112)
    lines.append("欄位說明")
    lines.append("=" * 112)
    lines.append("train/test_g/test_f/other : train/good, test/good, test/defect, other_fake 各自張數")
    lines.append("epoch                     : 被選為 best ckpt 的 epoch (來自 results.csv)")
    lines.append("cls_thr                   : image-level 分類門檻，決定 accuracy 等指標；fixed/percentile")
    lines.append("                            沿用驗證集校準值，oracle_f1/oracle_acc 直接在這批 test 上搜")
    lines.append("pix_thr                   : pixel-level 門檻，只給 visualize_all 的 predict_mask 二值化用，")
    lines.append("                            固定 F1 準則，跟 cls_thr 不同單位、不能互換")
    lines.append("acc/bacc/prec/rec_f/rec_g/f1 : final_test 在展開後的 test (good vs defect+other_fake) 上，")
    lines.append("                            用 cls_thr 算出來的分類指標")
    lines.append("I-AUROC/P-AUROC/P-PRO     : 同一次 final_test，門檻無關的排序/分割指標")
    lines.append("AP@.5                     : Detection AP@IoU0.5 (bbox)，整個 group 一個數字的物件偵測")
    lines.append("                            標準指標，不是單張圖的分數；<group_id>_bbox_top10/ 底下的 box 配對 F1 才是單張圖分數")
    lines.append("Miss/FalseAl              : Miss Rate/False Alarm，寬鬆判準——predict box 跟 GT box 雙向")
    lines.append("                            coverage 都 >= 50% 才算配對成功 (不用框得準，但框太大/太偏配不上")
    lines.append("                            一樣算錯)；漏檢的 GT 算 Miss，配不上任何 GT 的多餘框算 False Alarm，")
    lines.append("                            重疊的重複框只算一次，整個 group 池化成一個數字")
    lines.append("")

    # ── 每個 group 的細節 (混淆矩陣) ──────────────────────────
    lines.append("=" * 112)
    lines.append("各 group 混淆矩陣 (final_test，展開後的 test)")
    lines.append("=" * 112)
    for gid in gids:
        report = reports[gid]
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
