#!/bin/bash
#
# DeepPCB — 依序訓練所有 group (PCB 佈局)。
#
# 機制: main.py 的 `dataset` command 本來就支援多個 -d，會在同一個
# process 裡依序跑過每個 class (跟原版 run-mvtec.sh 列 15 個 MVTec 類別
# 是同一招)。每個 group 會拿到全新初始化的模型，ckpt 各自存在
# results_path/models/backbone_0/mvtec_<group_id>/ 底下，彼此不共用權重。
#
# 內建「跳過已完成」: trainer() 一開始會檢查該 group 是否已有
# ckpt_best_*，有的話直接跳過訓練。所以這支 script 中斷後重跑，
# 已經練完的 group 不會重練，只會接著跑剩下的。
#
# 12000 只有 14 張訓練圖 (其餘 group 都有 70+ 張)，預設排除，
# 有需要就自己加回 classes 陣列。
#
# 時間預估: 640 epochs x 10 個 group，會跑非常久 (單一 group 640 epoch
# 大約數小時，10 個 group 抓一整天以上)。想先看趨勢的話，
# 建議先把下面的 --meta_epochs 降到 50~100 探路，確認參數方向對了
# 再拉長。
#
datapath=/home/yuyun/Desktop/Innoserve/deeppcb_mice
classes=('00041' '12100' '12300' '13000' '20085' '44000' '50600' '77000' '90100' '92000')

flags=($(for class in "${classes[@]}"; do echo '-d '"${class}"; done))

cd ..

python main.py \
    --gpu 0 \
    --seed 0 \
    --test ckpt \
    --results_path results/0828/pcb_groups_k=0.25 \
    --visualize_all \
    --min_box_area 300 \
  net \
    -b wideresnet50 \
    -le layer2 \
    -le layer3 \
    --pretrain_embed_dimension 1536 \
    --target_embed_dimension 1536 \
    --patchsize 3 \
    --meta_epochs 1000 \
    --eval_epochs 5 \
    --dsc_layers 2 \
    --dsc_hidden 1024 \
    --pre_proj 1 \
    --k 0.5 \
    --n_neighbors 9 \
    --tangent_ratio 0.2 \
    --limit 1000 \
    --top_k 1 \
    --thr_mode oracle_acc \
    --accum_images 8 \
  dataset \
    --batch_size 1 \
    --resize 640 \
    --imagesize 640 "${flags[@]}" mvtec $datapath 
  

python summarize_results.py --results_path "results/0828/pcb_groups_k=0.25"

# python main.py \
#     --gpu 0 \
#     --seed 0 \
#     --test ckpt \
#     --results_path results/pcb_groups/k=0.5_t=0.2 \
#   net \
#     -b wideresnet50 \
#     -le layer2 \
#     -le layer3 \
#     --pretrain_embed_dimension 1536 \
#     --target_embed_dimension 1536 \
#     --patchsize 3 \
#     --meta_epochs 1000 \
#     --eval_epochs 5 \
#     --dsc_layers 2 \
#     --dsc_hidden 1024 \
#     --pre_proj 1 \
#     --k 0.5 \
#     --n_neighbors 9 \
#     --tangent_ratio 0.2 \
#     --limit 1000 \
#     --top_k 1 \
#     --thr_mode oracle_f1 \
#   dataset \
#     --batch_size 8 \
#     --resize 640 \
#     --imagesize 640 "${flags[@]}" mvtec $datapath 

# python main.py \
#     --gpu 0 \
#     --seed 0 \
#     --test ckpt \
#     --results_path results/pcb_groups/t=0.5_k=0.25 \
#   net \
#     -b wideresnet50 \
#     -le layer2 \
#     -le layer3 \
#     --pretrain_embed_dimension 1536 \
#     --target_embed_dimension 1536 \
#     --patchsize 3 \
#     --meta_epochs 1000 \
#     --eval_epochs 5 \
#     --dsc_layers 2 \
#     --dsc_hidden 1024 \
#     --pre_proj 1 \
#     --k 0.25 \
#     --n_neighbors 9 \
#     --tangent_ratio 0.5 \
#     --limit 1000 \
#     --top_k 1 \
#     --thr_mode oracle_f1 \
#   dataset \
#     --batch_size 8 \
#     --resize 640 \
#     --imagesize 640 "${flags[@]}" mvtec $datapath 

# python main.py \
#     --gpu 0 \
#     --seed 0 \
#     --test ckpt \
#     --results_path results/pcb_groups/t=0.05_k=0.25 \
#   net \
#     -b wideresnet50 \
#     -le layer2 \
#     -le layer3 \
#     --pretrain_embed_dimension 1536 \
#     --target_embed_dimension 1536 \
#     --patchsize 3 \
#     --meta_epochs 1000 \
#     --eval_epochs 5 \
#     --dsc_layers 2 \
#     --dsc_hidden 1024 \
#     --pre_proj 1 \
#     --k 0.25 \
#     --n_neighbors 9 \
#     --tangent_ratio 0.05 \
#     --limit 1000 \
#     --top_k 1 \
#     --thr_mode oracle_f1 \
#   dataset \
#     --batch_size 8 \
#     --resize 640 \
#     --imagesize 640 "${flags[@]}" mvtec $datapath 


