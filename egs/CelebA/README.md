# CelebA Example


```
cd ./u_net/
```

## 0. Preparation
Download CelebA dataset from [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

You need "In-The-Wild Images", so download `img_celeba.7z`.

After downloading, unzip `img_celeba.7z` like:

```
7z x "/content/img_celeba_7z/img_celeba.7z.001"
```

In Google Colaboratory evironment, this command may take few hours.
I do NOT recommend unzip on the mounted google drive. It takes much more hours and you cannot finish it in 12-hour limitaion.
You can use `download.ipynb`, if you alse run these scripts on Google Colaboratory.

You also need annotaions for training, validation, and test.
Annotations are expected same as the `list_bbox_celeba.txt` in original CelebA dataset.

You can use `prepare.ipynb` for making small size CelebA dataset.
I used 10,000 images for training, 1,000 for validation, and 5,000 for test.
If you change the number of images, you must pay attention image indices.


## 1. Training
You have to set proper path in `train.sh` for dataset.

```
image_root="../../../dataset/CelebA/img_celeba"
train_path="../../../dataset/CelebA/annotations/train.txt"
valid_path="../../../dataset/CelebA/annotations/valid.txt"
```

Then, 
```
. ./train.sh <OUTPUT_DIRECTORY>
```

If you want to resume training, 
```
. ./train.sh <OUTPUT_DIRECTORY> <MODEL_PATH>
```

You can change model configuration by changing patameters in `train.sh`.

## 2. Evaluation
You have to set proper path in `eval.sh` for dataset.

```
image_root="../../../dataset/CelebA/img_celeba"
test_path="../../../dataset/CelebA/annotations/test.txt"
```

Then,
```
. ./eval.sh <OUTPUT_DIRECTORY> <MODEL_PATH>
```

You can change model configuration by changing patameters in `eval.sh`.
The parameters shold match configuration of `<MODEL_PATH>`.

## 3. Demo
```
. ./demo.sh
```

