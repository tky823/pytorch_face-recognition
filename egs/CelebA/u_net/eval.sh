#!/bin/bash

. ./path.sh

exp_dir=$1
model_path=$2

image_root="../../../dataset/CelebA/img_celeba" # "/content/drive/My Drive/Colab Notebooks/db/CelebA/img_celeba"
test_path="../../../dataset/CelebA/annotations/test.txt" # "/content/drive/My Drive/Colab Notebooks/db/CelebA/annotations/test.txt"

H=256
W=256

K=2
n_candidate=10

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
separable=1
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

tag="UNet_H${H}_W${W}_G${G}_R${R}_dilated${dilated}_separable${separable}_${pool_backbone}-pool_${heatmap_loss}_${importance}_${optimizer}${lr}_b${batch_size}_e${epochs}"

if [ -z "${exp_dir}" ]; then
    exp_dir="./exp"
fi

save_dir="${exp_dir}/${tag}"
eval_dir="${save_dir}/eval"

if [ ! -e "${save_dir}/log" ]; then
    mkdir -p "${save_dir}/log"
fi

time_stamp=`TZ=UTC-9 date "+%Y%m%d-%T"`

eval_center_net.py \
--test_image_root "${image_root}" \
--test_path "${test_path}" \
--H ${H} \
--W ${W} \
--K ${K} \
--n_candidate ${n_candidate} \
--G ${G} \
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
--eval_dir "${eval_dir}" \
--model_path "${model_path}" | tee "${save_dir}/log/eval_${time_stamp}.log"
