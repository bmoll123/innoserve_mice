#!/bin/bash
#
# DeepPCB — 印刷電路板瑕疵 (open/short/mousebite/spur/copper/pin-hole)
# 資料由 DeepPCB/to_mice.py 轉出
#
#   train/good  1000  (trainval 的 _temp，無瑕疵範本)
#   test/good    500  (test 的 _temp)
#   test/defect  500  (test 的 _test，每張 1~15 個瑕疵)
#   ground_truth/defect  bbox pseudo mask
#
# 參數依實測的瑕疵尺寸挑選:
#   瑕疵框長邊 median 38 px (p25 34 / p75 46)，相對於 640x640。
#   --imagesize 640   維持原生解析度，不縮小 (縮到 288 的話 38px 只剩 17px)
#   -le layer2/layer3 layer2 stride 8 -> 38px 約 5 個 patch，夠用
#                     若要抓更小的 pin-hole 可改 -le layer1 -le layer2 (但要降 batch)
#   --top_k 60        640 -> layer2 grid 80x80 = 6400 patch，取最高 ~1%
#   --batch_size 4    6400 patch/張，batch 4 約 25600 patch
#
# 重跑前記得: rm -rf results/pcb/models
#
datapath=/home/yuyun/Desktop/Innoserve/deeppcb_mice
classes=('pcb')

flags=($(for class in "${classes[@]}"; do echo '-d '"${class}"; done))

cd ..
python main.py \
    --gpu 0 \
    --seed 0 \
    --test ckpt \
    --results_path results/pcb \
  net \
    -b wideresnet50 \
    -le layer2 \
    -le layer3 \
    --pretrain_embed_dimension 1536 \
    --target_embed_dimension 1536 \
    --patchsize 3 \
    --meta_epochs 640 \
    --eval_epochs 5 \
    --dsc_layers 2 \
    --dsc_hidden 1024 \
    --pre_proj 1 \
    --k 0.25 \
    --n_neighbors 9 \
    --tangent_ratio 0.2 \
    --limit 1000 \
    --top_k 1 \
    --dsc_margin 0.8 \
    --thr_mode fixed \
  dataset \
    --batch_size 8 \
    --resize 640 \
    --imagesize 640 "${flags[@]}" mvtec $datapath
