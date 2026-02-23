#!/bin/bash

datapath=/home/undergraduate/liwei/GLASS/cqy/dataset/Real-IAD

classes=("audiojack" "bottle_cap" "button_battery" "end_cap" "eraser" \
        "fire_hood" "mint" "mounts" "pcb" "phone_battery" "plastic_nut" \
        "plastic_plug" "porcelain_doll" "regulator" "rolled_strip_base" \
        "sim_card_set" "switch" "tape" "terminalblock" "toothbrush" \
        "toy" "toy_brick" "transistor1" "u_block" "usb" "usb_adaptor" \
        "vcpill" "wooden_beads" "woodstick" "zipper")
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
    --dsc_hidden 1024 \
    --pre_proj 1 \
    --k 0.25 \
    --n_neighbors 9 \
    --tangent_ratio 0.2 \
    --limit 392 \
  dataset \
    --batch_size 8 \
    --resize 288 \
    --imagesize 288 "${flags[@]}" real_iad $datapath