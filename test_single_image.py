"""
單張圖快速測試: 載入指定 group 的 checkpoint，對一張圖跑推論，用不同
min_box_area 過濾雜訊框，並排存成比較圖，方便決定要調到多少。

用法:
  python test_single_image.py --results_path results/0829/pcb_groups_k=0.25 \
      --group 12100 --stem 163 --source other_fake \
      --min_box_areas 25 50 100 200
"""
import argparse
import glob

import cv2
import numpy as np
import torch
import PIL.Image
from torchvision import transforms

import backbones
import common
import mice
import utils

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def mask_to_boxes(mask_bin, min_area, clean=True):
    m = mask_bin.astype(np.uint8) * 255
    if clean:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    return [b for b in boxes if b[2] * b[3] >= min_area]


def draw_boxes(img, boxes, color):
    out = img.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_path", required=True)
    ap.add_argument("--data_path", default="/home/yuyun/Desktop/Innoserve/deeppcb_mice")
    ap.add_argument("--group", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--source", default="other_fake", choices=["test/good", "test/defect", "other_fake"])
    ap.add_argument("--layers", nargs="+", default=["layer1", "layer2", "layer3"])
    ap.add_argument("--patchsize", type=int, default=3)
    ap.add_argument("--embed_dim", type=int, default=1536)
    ap.add_argument("--resize", type=int, default=640)
    ap.add_argument("--imagesize", type=int, default=640)
    ap.add_argument("--min_box_areas", type=int, nargs="+", default=[25, 50, 100, 200])
    ap.add_argument("--out", default="results/testing")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    backbone = backbones.load("wideresnet50")
    backbone.name, backbone.seed = "wideresnet50", None
    model = mice.MICE(device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=args.layers,
        device=device,
        input_shape=(3, args.imagesize, args.imagesize),
        pretrain_embed_dimension=args.embed_dim,
        target_embed_dimension=args.embed_dim,
        patchsize=args.patchsize,
        pre_proj=1,
        dsc_layers=2,
        dsc_hidden=1024,
        thr_mode="oracle_acc",
    )
    model.to(device)
    model.set_model_dir(
        f"{args.results_path}/models/backbone_0", f"mvtec_{args.group}", args.results_path
    )

    ckpt_paths = glob.glob(model.ckpt_dir + "/ckpt_best*")
    if not ckpt_paths:
        raise SystemExit(f"找不到 checkpoint: {model.ckpt_dir}/ckpt_best*")
    state_dict = torch.load(ckpt_paths[0], map_location=device)
    if "discriminator" in state_dict:
        model.discriminator.load_state_dict(state_dict["discriminator"])
        if "pre_projection" in state_dict:
            model.pre_projection.load_state_dict(state_dict["pre_projection"])
    else:
        model.load_state_dict(state_dict, strict=False)
    print(f"載入 checkpoint: {ckpt_paths[0]}")

    img_path = f"{args.data_path}/{args.group}/{args.source}/{args.stem}.jpg"
    mask_sub = {"test/good": None, "test/defect": "ground_truth/defect",
               "other_fake": "other_fake_masks"}[args.source]
    mask_path = f"{args.data_path}/{args.group}/{mask_sub}/{args.stem}.png" if mask_sub else None

    img_tf = transforms.Compose([
        transforms.Resize(args.resize), transforms.CenterCrop(args.imagesize),
        transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    mask_tf = transforms.Compose([
        transforms.Resize(args.resize), transforms.CenterCrop(args.imagesize), transforms.ToTensor(),
    ])

    pil_img = PIL.Image.open(img_path).convert("RGB")
    img_t = img_tf(pil_img).unsqueeze(0)

    model.forward_modules.eval()
    if model.pre_proj > 0:
        model.pre_projection.eval()
    model.discriminator.eval()
    with torch.no_grad():
        score, seg = model._predict(img_t)
    seg = seg[0]
    orig = utils.torch_format_2_numpy_img(img_t[0].cpu().numpy())

    if mask_path:
        import os
        if os.path.exists(mask_path):
            gt = PIL.Image.open(mask_path).convert("L")
            gt_arr = (mask_tf(gt).numpy()[0] * 255).astype(np.uint8)
        else:
            gt_arr = np.zeros(seg.shape, dtype=np.uint8)
    else:
        gt_arr = np.zeros(seg.shape, dtype=np.uint8)

    # 這張圖自己的 pixel 門檻 (跟 mice.py final_test 一樣的 F1 準則，只用這張圖)
    gt_bin = gt_arr > 0
    if gt_bin.any():
        from metrics import search_best_threshold
        pix_thr = search_best_threshold(seg.ravel(), gt_bin.ravel().astype(int), criterion="f1")["threshold"]
    else:
        pix_thr = float(np.percentile(seg, 99))
    print(f"score={float(np.asarray(score).ravel()[0]):.4f}  pixel_threshold={pix_thr:.4f}")

    pred_bin = (seg - pix_thr) > 0
    gt_boxes = mask_to_boxes(gt_bin, min_area=1, clean=False)

    RED, GREEN = (0, 0, 255), (0, 255, 0)
    cell = (256, 256)
    panels = [cv2.resize(draw_boxes(orig, gt_boxes, GREEN), cell)]
    labels = ["GT"]
    for min_area in args.min_box_areas:
        pred_boxes = mask_to_boxes(pred_bin, min_area=min_area, clean=True)
        panel = draw_boxes(draw_boxes(orig, gt_boxes, GREEN), pred_boxes, RED)
        panels.append(cv2.resize(panel, cell))
        labels.append(f"min_area={min_area} ({len(pred_boxes)} box)")
        print(f"min_box_area={min_area}: {len(pred_boxes)} 個 predict box")

    row = np.hstack(panels)
    import os
    os.makedirs(args.out, exist_ok=True)
    out_path = f"{args.out}/{args.group}_{args.source.replace('/', '_')}{args.stem}_minarea_compare.png"
    cv2.imwrite(out_path, row)
    print(f"存到 {out_path}  (由左到右: {', '.join(labels)})")


if __name__ == "__main__":
    main()
