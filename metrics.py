from sklearn import metrics
from skimage import measure

import cv2
import numpy as np
import pandas as pd


def compute_best_pr_re(anomaly_ground_truth_labels, anomaly_prediction_weights):
    """
    Computes the best precision, recall and threshold for a given set of
    anomaly ground truth labels and anomaly prediction weights.
    """
    precision, recall, thresholds = metrics.precision_recall_curve(anomaly_ground_truth_labels, anomaly_prediction_weights)
    f1_scores = 2 * (precision * recall) / (precision + recall)

    best_threshold = thresholds[np.argmax(f1_scores)]
    best_precision = precision[np.argmax(f1_scores)]
    best_recall = recall[np.argmax(f1_scores)]
    print(best_threshold, best_precision, best_recall)

    return best_threshold, best_precision, best_recall


def compute_imagewise_retrieval_metrics(anomaly_prediction_weights, anomaly_ground_truth_labels, path='training'):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).
    """
    auroc = metrics.roc_auc_score(anomaly_ground_truth_labels, anomaly_prediction_weights)
    ap = 0. if path == 'training' else metrics.average_precision_score(anomaly_ground_truth_labels, anomaly_prediction_weights)

    return {"auroc": auroc, "ap": ap}


def compute_classification_metrics(anomaly_prediction_weights, anomaly_ground_truth_labels, threshold=0.5):
    """
    在給定門檻下把 anomaly score 轉成硬分類，回傳 accuracy 等指標。

    正類 (label=1) = 假鞋 / 異常。
    因為 test set 通常類別不平衡 (例如 air_force 是 321 真 vs 518 假，
    全猜假就有 61.7% acc)，所以同時回傳 balanced_acc 與各類別的 recall。
    """
    scores = np.asarray(anomaly_prediction_weights, dtype=float).ravel()
    labels = np.asarray(anomaly_ground_truth_labels).astype(int).ravel()
    preds = (scores >= threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    n = len(labels)
    recall_fake = tp / (tp + fn) if (tp + fn) else 0.       # 假鞋抓出來的比例
    recall_real = tn / (tn + fp) if (tn + fp) else 0.       # 真鞋沒被冤枉的比例
    precision = tp / (tp + fp) if (tp + fp) else 0.
    f1 = 2 * precision * recall_fake / (precision + recall_fake) if (precision + recall_fake) else 0.

    return {
        "threshold": float(threshold),
        "acc": (tp + tn) / n if n else 0.,
        "balanced_acc": (recall_fake + recall_real) / 2,
        "precision": precision,
        "recall_fake": recall_fake,
        "recall_real": recall_real,
        "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": n,
    }


def search_best_threshold(anomaly_prediction_weights, anomaly_ground_truth_labels, criterion="balanced_acc"):
    """
    在 test set 上掃描門檻，找出讓 criterion 最大的那個。

    注意: 這是用 test 標籤挑出來的門檻，屬於 oracle 上界，會樂觀高估。
    要報告可部署的數字請看固定門檻 0.5 的版本。
    """
    scores = np.asarray(anomaly_prediction_weights, dtype=float).ravel()
    candidates = np.unique(scores)
    if len(candidates) > 500:
        candidates = np.quantile(candidates, np.linspace(0, 1, 500))

    best = None
    for t in candidates:
        m = compute_classification_metrics(scores, anomaly_ground_truth_labels, threshold=t)
        if best is None or m[criterion] > best[criterion]:
            best = m
    return best


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks, path='train',
                                        max_pixels=5_000_000, seed=0):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    像素數超過 max_pixels 時會隨機抽樣再計算。理由: DeepPCB 這種
    1000 張 640x640 的 test set 攤平後是 4.1 億個像素，roc_auc_score 會把它
    轉成 float64 再排序 —— 光是這一步就要 20GB RAM 和數分鐘，而且每個
    eval epoch 都要來一次。隨機抽樣對 AUROC 是不偏估計，500 萬個樣本的
    標準誤大約在小數點第三位，實務上跟全量算沒有差別。
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    if max_pixels and flat_anomaly_segmentations.size > max_pixels:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, flat_anomaly_segmentations.size, max_pixels)
        flat_anomaly_segmentations = flat_anomaly_segmentations[idx]
        flat_ground_truth_masks = flat_ground_truth_masks[idx]

    labels = flat_ground_truth_masks.astype(int)
    if len(np.unique(labels)) < 2:
        return {"auroc": 0., "ap": 0.}

    scores = flat_anomaly_segmentations.astype(np.float32)
    auroc = metrics.roc_auc_score(labels, scores)
    ap = 0. if path == 'training' else metrics.average_precision_score(labels, scores)

    return {"auroc": auroc, "ap": ap}


def search_best_pixel_threshold(segmentations, masks, criterion="f1", max_pixels=2_000_000, seed=0):
    """
    像 search_best_threshold，但在「攤平後的像素」上搜尋，不是影像分數。

    不能直接拿 image-level 的 best_threshold/oracle_f1 門檻套用在像素上:
    影像分數是該圖所有 patch 取 max，天生就比大多數像素分數高出一截，
    用影像門檻卡像素等於要求「每個像素都跟全圖最高分一樣高」，
    結果幾乎必定整片背景 (實測驗證過: 一張真的有瑕疵的圖，套用
    image-level oracle_f1=1.0 當像素門檻，predict mask 全黑)。

    像素數量通常有幾百萬到幾億，逐一過 500 個候選門檻太慢，
    所以先做隨機抽樣 (對門檻搜尋的影響可忽略，抽樣後再交給
    search_best_threshold 本身的候選門檻抽樣機制)。
    """
    if isinstance(segmentations, list):
        segmentations = np.stack(segmentations)
    if isinstance(masks, list):
        masks = np.stack(masks)

    scores = segmentations.ravel().astype(np.float32)
    labels = masks.ravel().astype(int)

    if len(np.unique(labels)) < 2:
        return {"threshold": 0.5, "f1": 0., "balanced_acc": 0., "acc": 0.}

    if scores.size > max_pixels:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, scores.size, max_pixels)
        scores, labels = scores[idx], labels[idx]

    return search_best_threshold(scores, labels, criterion=criterion)


def _box_iou(box1, box2):
    """box = (x, y, w, h)。"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    ix1, iy1 = max(x1, x2), max(y1, y2)
    ix2, iy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.


def compute_detection_ap(predictions, gts_per_image, iou_thresh=0.5):
    """
    標準物件偵測 AP (VOC2012 風格，all-point interpolation)，整個 group 一個
    數字，不是單張圖的分數。

    predictions: [(img_idx, box(x,y,w,h), confidence), ...]，全部圖的 predict
                 box 混在一起，不分圖。
    gts_per_image: {img_idx: [box(x,y,w,h), ...]}，每張圖各自的 GT box。

    做法: 依 confidence 由高到低排序，逐一跟同一張圖裡「還沒被配過」的 GT box
    配對 (取 IoU 最高的那個)，IoU >= iou_thresh 才算 TP，一個 GT 只能配一次
    (配過的下一個 predict box 再撞到只能算 FP，不能重複計分，這是標準做法，
    避免同一個瑕疵被框好幾次就灌水)。算出 precision-recall 曲線後取面積。
    """
    total_gt = sum(len(v) for v in gts_per_image.values())
    if total_gt == 0 or not predictions:
        return 0.

    preds = sorted(predictions, key=lambda p: -p[2])
    matched = {k: [False] * len(v) for k, v in gts_per_image.items()}
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))

    for i, (img_idx, box, _conf) in enumerate(preds):
        gts = gts_per_image.get(img_idx, [])
        best_iou, best_j = 0., -1
        for j, gt_box in enumerate(gts):
            if matched[img_idx][j]:
                continue
            iou = _box_iou(box, gt_box)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh:
            tp[i] = 1
            matched[img_idx][best_j] = True
        else:
            fp[i] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / total_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)

    mrec = np.concatenate([[0.], recall, [1.]])
    mpre = np.concatenate([[0.], precision, [0.]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _box_coverage(box_a, box_b):
    """box = (x, y, w, h)。回傳 intersection / box_a 面積。"""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a_area = aw * ah
    return inter / a_area if a_area > 0 else 0.


def _boxes_touch(box_a, box_b):
    """任何程度的重疊 (交集面積 > 0) 就算碰到。"""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    return (ix2 - ix1) > 0 and (iy2 - iy1) > 0


def _cluster_indices(indices, boxes):
    """把 indices 裡彼此有重疊 (碰到就算) 的分群，回傳 list of index-list。"""
    idx_list = list(indices)
    parent = {i: i for i in idx_list}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(idx_list)):
        for j in range(i + 1, len(idx_list)):
            if _boxes_touch(boxes[idx_list[i]], boxes[idx_list[j]]):
                union(idx_list[i], idx_list[j])

    clusters = {}
    for i in idx_list:
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


def _union_box(boxes, indices):
    """把一群框的外接矩形聯集起來，當作這群的代表框 (畫圖用，比只挑一個
    更能表達「這坨雜訊實際涵蓋的範圍」)。"""
    xs1 = [boxes[i][0] for i in indices]
    ys1 = [boxes[i][1] for i in indices]
    xs2 = [boxes[i][0] + boxes[i][2] for i in indices]
    ys2 = [boxes[i][1] + boxes[i][3] for i in indices]
    x1, y1, x2, y2 = min(xs1), min(ys1), max(xs2), max(ys2)
    return (x1, y1, x2 - x1, y2 - y1)


def match_miss_false_alarm(pred_boxes, gt_boxes, thresh=0.3):
    """
    回傳 (tp, fp, fn, matched_pred_boxes, fp_boxes) 給 miss rate / false alarm
    rate 用，也給畫圖用。

    跟 compute_detection_ap 的單向 IoU 配對不同，這裡刻意用寬鬆、雙向的
    coverage 判準:「predict box 覆蓋 GT 面積 >= thresh」且「GT 覆蓋 predict
    box 面積 >= thresh」同時成立才算配對候選 (兩邊都要夠，框太大或太小單獨
    滿足一邊都不算——避免一個超大框隨便掃過去就騙過判準)。

    逐一挑出目前品質最高 (兩方向 coverage 的較小值最大) 的候選配對當贏家，
    清掉跟贏家有重疊 (碰到就算，不看重疊比例) 的其他 predict box —— 當作
    同一個偵測的重複框，不計入 false alarm；但如果某個框對另一個「還沒
    配對的 GT」自己也是合格候選，就保留給下一輪配對，不能清掉。重複整個
    流程直到沒有合格候選為止。

    最後剩下、沒配到任何 GT 的框，如果彼此還有重疊也要合併成一個 (聯集框)，
    不能各自獨立算一次 false alarm (同一坨雜訊不該被拆成好幾筆誤報)。

    matched_pred_boxes: 配對成功的贏家框列表。
    fp_boxes: 配不到任何 GT、重疊的已合併成聯集框的誤報框列表。
    """
    remaining_pred = set(range(len(pred_boxes)))
    unmatched_gt = set(range(len(gt_boxes)))
    tp = 0
    matched_pred_boxes = []

    def is_candidate(pi, gi):
        p, g = pred_boxes[pi], gt_boxes[gi]
        return _box_coverage(g, p) >= thresh and _box_coverage(p, g) >= thresh

    def quality(pi, gi):
        p, g = pred_boxes[pi], gt_boxes[gi]
        return min(_box_coverage(g, p), _box_coverage(p, g))

    while True:
        best = max(
            ((quality(pi, gi), pi, gi) for gi in unmatched_gt for pi in remaining_pred
             if is_candidate(pi, gi)),
            default=None,
        )
        if best is None:
            break
        _, win_pi, win_gi = best
        tp += 1
        matched_pred_boxes.append(pred_boxes[win_pi])
        unmatched_gt.discard(win_gi)
        remaining_pred.discard(win_pi)

        for pi in list(remaining_pred):
            if _boxes_touch(pred_boxes[pi], pred_boxes[win_pi]):
                still_useful = any(is_candidate(pi, gi) for gi in unmatched_gt)
                if not still_useful:
                    remaining_pred.discard(pi)

    fn = len(unmatched_gt)
    fp_clusters = _cluster_indices(remaining_pred, pred_boxes)
    fp_boxes = [_union_box(pred_boxes, cluster) for cluster in fp_clusters]
    return tp, len(fp_boxes), fn, matched_pred_boxes, fp_boxes


def compute_pro(masks, amaps, num_th=200):
    # 每個門檻的結果先收在 list，最後一次組成 DataFrame。
    # 原本用 df.append 逐列累加，那個 API 已被 pandas 棄用，每次呼叫都會噴
    # FutureWarning (200 個門檻 = 200 行警告)，而且逐列 append 是 O(n^2)。
    rows = []
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            binary_amap = cv2.dilate(binary_amap.astype(np.uint8), k)
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        rows.append({"pro": np.mean(pros), "fpr": fpr, "threshold": th})

    df = pd.DataFrame(rows, columns=["pro", "fpr", "threshold"])
    df = df[df["fpr"] < 0.3]
    df["fpr"] = (df["fpr"] - df["fpr"].min()) / (df["fpr"].max() - df["fpr"].min() + 1e-10)

    pro_auc = metrics.auc(df["fpr"], df["pro"])
    return pro_auc
