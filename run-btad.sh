#!/bin/bash

datapath=/home/undergraduate/liwei/GLASS/cqy/dataset/BTAD

classes=('01' '02' '03')
flags=($(for class in "${classes[@]}"; do echo '-d '"${class}"; done))

cd ..
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
    --dsc_hidden 1536 \
    --pre_proj 1 \
    --k 0.22 \
    --n_neighbors 9 \
    --tangent_ratio 0.3 \
    --limit 392 \
  dataset \
    --aug_path $augpath \
    --batch_size 8 \
    --resize 288 \
    --imagesize 288 "${flags[@]}" btad $datapath