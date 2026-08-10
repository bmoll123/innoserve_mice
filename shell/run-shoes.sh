#!/bin/bash
#
# 真假鞋資料集 (由 dataset_fake_shoes/to_mice.py 轉出的 MVTec-AD 格式)
#
# 注意: trainer() 一開始會檢查 ckpt_best*，找到就直接跳過訓練進入測試模式。
#       要重新訓練請先刪掉 results/models/。
#
datapath=/home/yuyun/Desktop/Innoserve/dataset_mice
classes=('air_force' 'jordan1')

flags=($(for class in "${classes[@]}"; do echo '-d '"${class}"; done))

cd ..
python main.py \
    --gpu 0 \
    --seed 0 \
    --test ckpt \
    --results_path results/0727 \
  net \
    -b wideresnet50 \
    -le layer2 \
    -le layer3 \
    --pretrain_embed_dimension 1536 \
    --target_embed_dimension 1536 \
    --patchsize 3 \
    --meta_epochs 200 \
    --eval_epochs 1 \
    --dsc_layers 2 \
    --dsc_hidden 1024 \
    --pre_proj 1 \
    --k 0.25 \
    --n_neighbors 9 \
    --tangent_ratio 0.2 \
    --limit 1400 \
    --top_k 13 \
  dataset \
    --batch_size 8 \
    --resize 288 \
    --imagesize 288 "${flags[@]}" mvtec $datapath
