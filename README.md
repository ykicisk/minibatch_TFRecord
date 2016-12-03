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

グラフはミニバッチを構成するデータの元ファイルの分布が、ミニバッチを進めていくとどう変化するかを表す。
縦軸はTFRecordファイル、横軸はミニバッチに含まれるデータ数(累積)、batch[0-9]は全ミニバッチを時間順に均等に分けたもの。

下の結果であれば、shuffle_batch始めの1/10のミニバッチには、file5, file6由来のデータしか含まれておらず、
file5に入っているデータは半分以上がshuffle_batch始めの1/10のミニバッチに入っていることがわかる。

![結果](images/result.png)
