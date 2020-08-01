#!/bin/bash

. ./path.sh

H=256
W=256
K=2

R=4
G=1

# Downsampling
K_down=3
S_down=2
pool_down='max'

# Backbone
channels=[64,128,256,512]
K_backbone=3
S_backbone=1
dilated=1
separable=0
pool_backbone='max'
nonlinear_backbone='relu'

# Criterion
heatmap_loss='bce'
importance='[1.0,1.0,1.0]'

batch_size=32
epochs=50

# Optimizer
optimizer='adam'
lr=0.001

n_candidate=10
confidence_threshold=0.5
iou_threshold=0.1

tag="UNet_H${H}_W${W}_G${G}_R${R}_dilated${dilated}_separable${separable}_${pool_backbone}-pool_${heatmap_loss}_${importance}_${optimizer}${lr}_b${batch_size}_e${epochs}"
save_dir="./exp/${tag}"
model_path="${save_dir}/model/best.pth"

demo.py \
--H ${H} \
--W ${W} \
--K ${K} \
--R ${R} \
--K_down ${K_down} \
--S_down ${S_down} \
--pool_down ${pool_down} \
--channels ${channels} \
--K_backbone ${K_backbone} \
--S_backbone ${S_backbone} \
--dilated ${dilated} \
--separable ${separable} \
--pool_backbone ${pool_backbone} \
--nonlinear_backbone ${nonlinear_backbone} \
--n_candidate ${n_candidate} \
--confidence_threshold ${confidence_threshold} \
--iou_threshold ${iou_threshold} \
--model_path ${model_path}
