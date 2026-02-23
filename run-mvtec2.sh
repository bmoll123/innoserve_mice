#!/bin/bash

datapath=/home/undergraduate/liwei/GLASS/cqy/dataset/mvtec_ad_2

classes=('can' 'fabric' 'fruit_jelly' 'rice' 'sheet_metal' 'vial' 'wallplugs' 'walnuts')
flags=($(for class in "${classes[@]}"; do echo '-d '"${class}"; done))

python main.py \
    --gpu 0 \
    --seed 0 \
    --test ckpt \
  net \
    -b wideresnet50 \
    -le layer2 \
    -le layer3 \
    --pretrain_embed_dimension 1536 \
    --target_embed_dimension 1536 \
    --patchsize 3 \
    --meta_epochs 640 \
    --eval_epochs 1 \
    --dsc_layers 2 \
    --dsc_hidden 1024 \
    --pre_proj 1 \
    --k 0.25 \
    --n_neighbors 9 \
    --tangent_ratio 0.2 \
  dataset \
    --batch_size 8 \
    --resize 288 \
    --imagesize 288 "${flags[@]}" mvtec2 $datapath
