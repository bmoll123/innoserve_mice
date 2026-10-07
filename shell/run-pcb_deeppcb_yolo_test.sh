#!/bin/bash
#
# 拿既有的 MICE checkpoint (results/0828/pcb_groups_k=0.25，沒有重練)，
# 對三份 deeppcb_defect_train_{1,3,5} 各自的 test 跑一次 final_test()，
# 跟 YOLO 那邊 run-deeppcb-yolo-supervised.sh 用同一批 test 圖比較。
#
# 輸出各自分開，不會互相覆蓋：
#   results/0918_deeppcb_yolo_test/defect_train_<n>/pcb_groups_k=0.25/
#
set -e

cd /home/yuyun/Desktop/Innoserve/MICE

for n in 1 3 5; do
  data_src="/home/yuyun/Desktop/Innoserve/deeppcb_defect_train_${n}"
  results_path="results/0918_deeppcb_yolo_test/defect_train_${n}/pcb_groups_k=0.25"

  echo
  echo "########################################"
  echo "## defect_train_count=${n}  (${data_src})"
  echo "########################################"

  python main.py \
      --gpu 0 --seed 0 --test ckpt \
      --results_path "$results_path" \
      --ckpt_source_path results/0828/pcb_groups_k=0.25 \
      --final_test_data_path "$data_src" \
      --dataset_layout yolo \
      --visualize_all --min_box_area 300 \
    net \
      -b wideresnet50 -le layer2 -le layer3 \
      --pretrain_embed_dimension 1536 --target_embed_dimension 1536 \
      --patchsize 3 --meta_epochs 1000 --eval_epochs 5 \
      --dsc_layers 2 --dsc_hidden 1024 --pre_proj 1 \
      --k 0.5 --n_neighbors 9 --tangent_ratio 0.2 --limit 1000 --top_k 1 \
      --thr_mode oracle_acc --accum_images 8 \
    dataset \
      --batch_size 8 --resize 640 --imagesize 640 \
      -d 00041 -d 12100 -d 12300 -d 13000 -d 20085 -d 44000 -d 50600 -d 77000 -d 90100 -d 92000 \
      mvtec /home/yuyun/Desktop/Innoserve/deeppcb_mice

  python summarize_results.py --results_path "$results_path"
done

echo
echo "== 三份資料集都跑完了 =="
echo "  results/0918_deeppcb_yolo_test/defect_train_1/pcb_groups_k=0.25"
echo "  results/0918_deeppcb_yolo_test/defect_train_3/pcb_groups_k=0.25"
echo "  results/0918_deeppcb_yolo_test/defect_train_5/pcb_groups_k=0.25"
