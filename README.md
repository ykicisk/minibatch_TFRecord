# minibatch_TFRecord

* TFRecordファイルを入力とした`tf.train.shuffle_batch`を行うスクリプトです。
* パラメータを変えたときの`tf.train.shuffle_batch`の偏りを調べることもできます。
	* ここで言う偏り＝同じミニバッチ内でどれ位同じTFRecordファイルに含まれるファイルが有るか

## dependencies

* python2.7
* tensorflow
* numpy
* matplotlib

## usage

### テスト用TFRecordファイルの保存

```sh
$ mkdir data
$ ./write_tfrecords.py data
```

### テスト用TFRecordを使ったshuffle_batch

```sh
$ ./mini_batch.py data
```

## result

デフォルトパラメータのときの結果。

グラフはミニバッチを構成するデータの元ファイルの分布の変化がどう変化するかを表す。
縦軸はTFRecordファイル、横軸はミニバッチに含まれるデータ数、batch[0-9]は全ミニバッチを時間順に均等に分けたもの。

![結果](images/result.png)

このパラメータでは、shuffle_batchをおこなっても、１バッチに含まれるデータの元ファイルは3つ程度であり。
shuffle_batchを行う際に、5, 6, 0, 3, 7, 8, 1, 2, 9, 4のような順番でファイルが読み込まれた事がわかる。
