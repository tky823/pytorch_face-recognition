#!/bin/bash

. ./path.sh

exp_dir=$1
continue_from=$2

image_root="../../../dataset/CelebA/img_celeba" # "/content/drive/My Drive/Colab Notebooks/db/CelebA/img_celeba"
train_path="../../../dataset/CelebA/annotations/train.txt" # "/content/drive/My Drive/Colab Notebooks/db/CelebA/annotations/train.txt"
valid_path="../../../dataset/CelebA/annotations/valid.txt" # "/content/drive/My Drive/Colab Notebooks/db/CelebA/annotations/valid.txt"

H=256
W=256

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

# Resume training
continue_from="${continue_from}"

tag="UNet_H${H}_W${W}_G${G}_R${R}_dilated${dilated}_separable${separable}_${pool_backbone}-pool_${heatmap_loss}_${importance}_${optimizer}${lr}_b${batch_size}_e${epochs}"

if [ -z "${exp_dir}" ]; then
    exp_dir="./exp"
fi

save_dir="${exp_dir}/${tag}"
model_dir="${save_dir}/model"
loss_dir="${save_dir}/loss"
sample_dir="${save_dir}/sample"

if [ ! -e "${save_dir}/log" ]; then
    mkdir -p "${save_dir}/log"
fi

time_stamp=`TZ=UTC-9 date "+%Y%m%d-%T"`

train_center_net.py \
--train_image_root "${image_root}" \
--valid_image_root "${image_root}" \
--train_path "${train_path}" \
--valid_path "${valid_path}" \
--H ${H} \
--W ${W} \
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
--heatmap_loss ${heatmap_loss} \
--importance ${importance} \
--batch_size ${batch_size} \
--epochs ${epochs} \
--optimizer ${optimizer} \
--lr ${lr} \
--model_dir "${model_dir}" \
--loss_dir "${loss_dir}" \
--sample_dir "${sample_dir}" \
--continue_from "${continue_from}" | tee "${save_dir}/log/train_${time_stamp}.log"
