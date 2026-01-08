#!/bin/bash

set -e

# ====== 这里三行是你可以根据需要调整的参数 ======
N_VAL=500        # 验证集图片数量
N_TRAIN=10000    # 训练集图片数量（你要的一万张）
NUM_THREADS=20   # 同时跑多少个 pdflatex 进程
# ==============================================

mkdir -p dataset/val/images dataset/val/labels
mkdir -p dataset/train/images dataset/train/labels
mkdir -p dataset/train/graphs dataset/val/graphs

### ---------------- VAL ---------------- ###
val_task() {
    i=$1
    mkdir -p "dataset/val/tmp_$i"
    cd "dataset/val/tmp_$i" || exit

    python3 ../../../gen.py "$i" > "$i.tex"
    pdflatex -interaction=nonstopmode "$i.tex"
    pdftoppm -png "$i.pdf" "$i"

    cd ..   # 回到 dataset/val/

    mv "tmp_$i/$i-1.png" "images/$i.png"
    mv "tmp_$i/$i.txt"  "labels/$i.txt"
    mv "tmp_$i/${i}_graph.json" "graphs/$i.json"

    rm -rf "tmp_$i"
}

num_threads=$NUM_THREADS
for i in $(seq 1 "$N_VAL"); do
    val_task "$i" &
    # 控制同时运行的后台任务数量
    while [ "$(jobs -r | wc -l)" -ge "$num_threads" ]; do
        sleep 1
    done
done
wait

### ---------------- TRAIN ---------------- ###
train_task() {
    i=$1
    mkdir -p "dataset/train/tmp_$i"
    cd "dataset/train/tmp_$i" || exit

    python3 ../../../gen.py "$i" > "$i.tex"
    pdflatex -interaction=nonstopmode "$i.tex"
    pdftoppm -png "$i.pdf" "$i"

    cd ..   # 回到 dataset/train/

    mv "tmp_$i/$i-1.png" "images/$i.png"
    mv "tmp_$i/$i.txt"  "labels/$i.txt"
    mv "tmp_$i/${i}_graph.json" "graphs/$i.json"

    rm -rf "tmp_$i"
}

num_threads=$NUM_THREADS
for i in $(seq 1 "$N_TRAIN"); do
    train_task "$i" &
    while [ "$(jobs -r | wc -l)" -ge "$num_threads" ]; do
        sleep 1
    done
done
wait
