# pytorch-face_recognition
A pytorch implementation of CenterNet based face recognition

# Example

```
cd ./egs/CelebA/u_net/
```

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

### Early stopping
In my implementation, if the model doesn't update the best loss (= smallest) in continuous 3 epochs, the learning rate will be halved. 
if the model doesn't in continuous 10 epochs, the training will be stopped before the training epoch reaches the number you specified.

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
