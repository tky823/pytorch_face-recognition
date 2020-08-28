# CelebAデータセットを用いた例


```
cd ./unet/
```

## 0. 事前準備
CelebA データセットを[ここ](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)からダウンロードする．

"In-The-Wild Images" が必要であるため，`img_celeba.7z` をダウンロードすることに注意．

その後，以下のようなコマンドで`img_celeba.7z`を解凍する．

```
7z x "/content/img_celeba_7z/img_celeba.7z.001"
```

Google Colaboratory環境を使っているのであれば，上記の実行には数時間かかる．
マウントされたGoogleドライブ上で解凍することは，すすめない．Google Colaboratoryの12時間の使用制限内に完了しないため．
Google Colaboratoryを使うのであれば， `download.ipynb`を参考にできる．

さらに，学習，検証，テストそれぞれのアノテーションが必要である．
アノテーションのファイルは，元々のCelebAデータセットの`list_bbox_celeba.txt`と同じ構造で記載すること．

`prepare.ipynb`を使うことで，CelebAデータセットの部分的な小さいサイズのデータを作成できる．
そのまま実行すれば，学習用に10,000枚，検証用に1,000枚，テスト用に5,000枚のアノテーションデータが作成される．
枚数を変更する場合は，画像のインデックスがかぶらないように注意すること．


## 1. 学習
`train.sh`中でデータセットへの適切なパスを指定する．

```
image_root="../../../dataset/CelebA/img_celeba"
train_path="../../../dataset/CelebA/annotations/train.txt"
valid_path="../../../dataset/CelebA/annotations/valid.txt"
```

その後，

```
. ./train.sh <OUTPUT_DIRECTORY>
```

学習を再開するときには，
```
. ./train.sh <OUTPUT_DIRECTORY> <MODEL_PATH>
```

`train.sh`中のパラメータを変更することで，ネットワーク構造を変えることができる．

### 学習の早期終了
この実装では，DNNの最小のロスが3エポック連続して更新されなかった場合，学習率を半分にしている．
10エポック連続して更新されなかった場合，指定したエポック数に到達する前に学習を早期終了させる．

## 2. 評価
`eval.sh`中でデータセットへの適切なパスを指定する．

```
image_root="../../../dataset/CelebA/img_celeba"
test_path="../../../dataset/CelebA/annotations/test.txt"
```

その後，

```
. ./eval.sh <OUTPUT_DIRECTORY> <MODEL_PATH>
```

 `eval.sh`中のパラメータを変更することで，ネットワーク構造を変えることができる．
 ただし，`<MODEL_PATH>`中の構造と一致するようにパラメータは指定すること．

## 3. リアルタイムデモ
```
. ./demo.sh
```

