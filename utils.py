import numpy as np

import csv
import os
import random
import torch
import shutil

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def create_storage_folder(
        main_folder_path, project_folder, group_folder, run_name, mode="iterate"
):
    os.makedirs(main_folder_path, exist_ok=True)
    save_path = main_folder_path
    return save_path


def set_torch_device(gpu_ids):
    """Returns correct torch.device.

    Args:
        gpu_ids: [list] list of gpu ids. If empty, cpu is used.
    """
    if len(gpu_ids):
        return torch.device("cuda:{}".format(gpu_ids[0]))
    return torch.device("cpu")


def fix_seeds(seed, with_torch=True, with_cuda=True):
    """Fixed available seeds for reproducibility.

    Args:
        seed: [int] Seed value.
        with_torch: Flag. If true, torch-related seeds are fixed.
        with_cuda: Flag. If true, torch+cuda-related seeds are fixed
    """
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)
    if with_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def compute_and_store_final_results(
        results_path,
        results,
        column_names=None,
        row_names=None,
):
    """Store computed results as CSV file.

    Args:
        results_path: [str] Where to store result csv.
        results: [List[List]] List of lists containing results per dataset,
                 with results[i][0] == 'dataset_name' and results[i][1:6] =
                 [instance_auroc, full_pixelwisew_auroc, full_pro,
                 anomaly-only_pw_auroc, anomaly-only_pro]
    """
    if row_names is not None:
        assert len(row_names) == len(results), "#Rownames != #Result-rows."

    mean_metrics = {}
    for i, result_key in enumerate(column_names):
        mean_metrics[result_key] = np.mean([x[i] for x in results])

    savename = os.path.join(results_path, "results.csv")
    with open(savename, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        header = column_names
        if row_names is not None:
            header = ["Row Names"] + header

        csv_writer.writerow(header)
        for i, result_list in enumerate(results):
            csv_row = result_list
            if row_names is not None:
                csv_row = [row_names[i]] + result_list
            csv_writer.writerow(csv_row)
        mean_scores = list(mean_metrics.values())
        if row_names is not None:
            mean_scores = ["Mean"] + mean_scores
        csv_writer.writerow(mean_scores)

    mean_metrics = {"mean_{0}".format(key): item for key, item in mean_metrics.items()}
    return mean_metrics


def del_remake_dir(path, del_flag=True):
    if os.path.exists(path):
        if del_flag:
            shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)


def torch_format_2_numpy_img(img):
    if img.shape[0] == 3:
        img = img.transpose([1, 2, 0])
        img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        img = img[:, :, [2, 1, 0]]
        img = (img * 255).astype('uint8')
    else:
        img = img.transpose([1, 2, 0])
        img = np.repeat(img, 3, axis=-1)
        img = (img * 255).astype('uint8')
    return img


def plot_confusion_matrix(cls, name, out_path):
    """畫 2x2 混淆矩陣 (good / fake)，存成 PNG。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 列 = 真實標籤, 欄 = 預測
    cm = np.array([[cls["tn"], cls["fp"]],
                   [cls["fn"], cls["tp"]]], dtype=int)
    labels = ["good", "fake"]

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(cm.max(), 1))

    for i in range(2):
        row_total = cm[i].sum()
        for j in range(2):
            pct = cm[i, j] / row_total * 100 if row_total else 0.
            ax.text(j, i, f"{cm[i, j]}\n{pct:.1f}%", ha="center", va="center",
                    fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{name}\nacc {cls['acc'] * 100:.2f}%  "
                 f"balanced acc {cls['balanced_acc'] * 100:.2f}%  "
                 f"(thr {cls['threshold']:.2f})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_eval_report(cls, name, img_paths, labels_gt, scores, threshold, out_path,
                      single_class=False, extra_metrics=None, pixel_threshold=None,
                      pixel_threshold_is_oracle=False):
    """把 test set 每張圖的預測結果寫成人看的 txt，判錯的排在最前面。

    extra_metrics: 選填，例如 {"I-AUROC": .., "P-AUROC": .., "P-PRO": ..}，
    門檻無關的排序/分割指標，印在分類指標下面。
    pixel_threshold: 選填，visualize_all 二值化 predict_mask 用的 pixel 門檻
    (跟 threshold 是不同單位，image-level vs pixel-level，不能共用)。
    pixel_threshold_is_oracle: pixel_threshold 是不是「每張圖各自拿自己的 GT
    反推出來的」平均值 —— 不是可部署的方法，只是這批圖各自的分數最多能標多準
    的上界，True 時會在報告裡明講，避免被誤會成一個固定、可重現的門檻。
    """
    rows = []
    for p, lab, sc in zip(img_paths, labels_gt, scores):
        lab = int(lab)
        pred = int(float(sc) >= threshold)
        rows.append({
            "file": os.path.basename(str(p)),
            "actual": "fake" if lab else "good",
            "pred": "fake" if pred else "good",
            "score": float(sc),
            "ok": pred == lab,
        })

    wrong = [r for r in rows if not r["ok"]]
    width = max([len(r["file"]) for r in rows] + [20])

    with open(out_path, "w") as f:
        f.write(f"Evaluation report: {name}\n")
        f.write("=" * 72 + "\n")
        f.write(f"threshold            : {threshold:.4f}\n")
        if pixel_threshold is not None:
            if pixel_threshold_is_oracle:
                f.write(f"pixel threshold      : {pixel_threshold:.4f}  "
                        f"(每張圖各自拿自己的 GT 反推出來的平均值，逐張不同、非固定門檻，"
                        f"不可部署，只是這批圖各自分數最多能標多準的上界)\n")
            else:
                f.write(f"pixel threshold      : {pixel_threshold:.4f}  "
                        f"(visualize_all predict_mask 二值化用，跟上面的 image-level threshold 不同單位)\n")
        f.write(f"total test images    : {len(rows)}\n")
        f.write(f"  actual good        : {sum(1 for r in rows if r['actual'] == 'good')}\n")
        f.write(f"  actual fake        : {sum(1 for r in rows if r['actual'] == 'fake')}\n")
        f.write(f"wrong predictions    : {len(wrong)}\n\n")

        if single_class:
            f.write("test set 只有一種標籤，accuracy / AUROC 無意義，以下僅供分數排序參考。\n\n")
        else:
            f.write("Confusion matrix (row = actual, col = predicted)\n")
            f.write(f"{'':>10}{'good':>8}{'fake':>8}\n")
            f.write(f"{'good':>10}{cls['tn']:>8}{cls['fp']:>8}\n")
            f.write(f"{'fake':>10}{cls['fn']:>8}{cls['tp']:>8}\n\n")
            f.write(f"accuracy             : {cls['acc'] * 100:.2f} %\n")
            f.write(f"balanced accuracy    : {cls['balanced_acc'] * 100:.2f} %\n")
            f.write(f"precision (fake)     : {cls['precision'] * 100:.2f} %\n")
            f.write(f"recall  fake / good  : {cls['recall_fake'] * 100:.2f} % / "
                    f"{cls['recall_real'] * 100:.2f} %\n")
            f.write(f"f1 (fake)            : {cls['f1'] * 100:.2f} %\n")
            if extra_metrics:
                for k, v in extra_metrics.items():
                    f.write(f"{k:<21}: {v * 100:.2f} %\n")
            f.write("\n")

        def dump(title, items):
            f.write("-" * 72 + "\n")
            f.write(f"{title}  ({len(items)})\n")
            f.write("-" * 72 + "\n")
            f.write(f"{'file':<{width}} {'actual':>7} {'pred':>7} {'score':>9}  result\n")
            for r in sorted(items, key=lambda x: -x["score"]):
                f.write(f"{r['file']:<{width}} {r['actual']:>7} {r['pred']:>7} "
                        f"{r['score']:>9.6f}  {'OK' if r['ok'] else 'WRONG'}\n")
            f.write("\n")

        if wrong:
            dump("WRONG PREDICTIONS", wrong)
        dump("ALL TEST IMAGES (score desc)", rows)
