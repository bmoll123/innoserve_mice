#!/bin/bash
#
# Travis Scott Jordan 1 Low "Tropical Pink"
# 資料由 Dataset/prep_jordan.py 轉出 (512x512 tile，無補邊、無裁切損失)
#
# 針對「瑕疵非常細微」做的調整，與 run-shoes.sh 的差異:
#   --imagesize 512   tile 本身就是 512，transform 等於不動它，畫質完整保留
#   -le layer1        改抽 layer1+layer2 (stride 4/8) 而非 layer2+layer3 (stride 8/16)
#                     淺層特徵空間解析度高，才看得到針腳等級的細節
#   --k 0.1           合成異常的擾動幅度調小 (0.25 -> 0.1)，逼 discriminator
#                     學一條更貼近正常流形的邊界，才分得出細微差異
#   --batch_size 2    512x512 + layer1 的 patch 數暴增 (128x128=16384/張)，必須降 batch
#   --top_k 40        patch 數變多，取分數最高的 ~0.25% 平均
#
# 注意: 每張原圖會切成數個 tile，predictions CSV 是「每個 tile 一列」。
#       要看單張影像的結果，把檔名 __t0/__t1... 前綴相同的取最大值。
#
# 重跑前記得刪掉舊 ckpt: rm -rf results/jordan/models
#
datapath=/home/yuyun/Desktop/Innoserve/Dataset/jordan_mice
classes=('tropical_pink')

flags=($(for class in "${classes[@]}"; do echo '-d '"${class}"; done))

cd ..
python main.py \
    --gpu 0 \
    --seed 0 \
    --test ckpt \
    --results_path results/jordan \
  net \
    -b wideresnet50 \
    -le layer1 \
    -le layer2 \
    --pretrain_embed_dimension 1536 \
    --target_embed_dimension 1536 \
    --patchsize 3 \
    --meta_epochs 200 \
    --eval_epochs 5 \
    --dsc_layers 2 \
    --dsc_hidden 1024 \
    --pre_proj 1 \
    --k 0.1 \
    --n_neighbors 9 \
    --tangent_ratio 0.2 \
    --limit 400 \
    --top_k 40 \
  dataset \
    --batch_size 2 \
    --resize 512 \
    --imagesize 512 "${flags[@]}" mvtec $datapath
