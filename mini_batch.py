#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse as ap
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def show_batch_bias(batch_contents, num_files, num_bins=10):
    freq_matrix = np.zeros((num_files, num_bins))
    for batch_idx, batch in enumerate(batch_contents):
        bin_idx = int(num_bins * batch_idx / len(batch_contents))
        for file_idx in batch:
            freq_matrix[file_idx, bin_idx] += 1
    plots = []

    ind = np.arange(num_files)
    height = 0.5

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for bin_idx in range(num_bins):
        if bin_idx == 0:
            bottom = np.zeros((num_files))
        else:
            bottom = np.sum(freq_matrix[:, :bin_idx], axis=1)
        color = cm.jet(1.0 * bin_idx / num_bins)
        plots.append(ax.barh(ind, freq_matrix[:, bin_idx], left=bottom,
                             color=color, height=height))
    plt.yticks(ind + height/2.,
               ['file{0:02d}'.format(i) for i in range(num_files)])

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.legend([p[0] for p in plots],
              ['batch{0}'.format(i) for i in range(num_files)],
              loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def minibatch(src_path, min_after_dequeue, batch_size, num_threads):
    filepaths = glob.glob("{src}/*.tfrecord".format(src=src_path))
    file_queue = tf.train.string_input_producer(filepaths,
                                                num_epochs=1,
                                                shuffle=True)

    reader = tf.TFRecordReader()
    key, selialized_data = reader.read(file_queue)

    features_def = {
        "file_idx": tf.FixedLenFeature([1], tf.int64),
        "record_idx": tf.FixedLenFeature([1], tf.int64),
        "data": tf.FixedLenFeature([2, 4], tf.float32)
    }
    features = tf.parse_single_example(selialized_data,
                                       features=features_def)

    capacity = min_after_dequeue + 3 * batch_size
    features_batch = tf.train.shuffle_batch(
        features,
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue,
        num_threads=num_threads,
        allow_smaller_final_batch=True
    )

    init_op = [tf.initialize_all_variables(),
               tf.initialize_local_variables()]

    batch_contents = []
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                sample = sess.run(features_batch)
                batch_contents.append(sample["file_idx"].flatten())
                # optimization or evaluation with sample["data"]
        except tf.errors.OutOfRangeError:
            pass
        finally:
            coord.request_stop()
            coord.join(threads)

    show_batch_bias(batch_contents, len(filepaths))


def main():
    description = """mini batch with TFRecord files

see: https://www.tensorflow.org/versions/r0.12/how_tos/reading_data/index.html
"""

    class Formatter(ap.ArgumentDefaultsHelpFormatter,
                    ap.RawDescriptionHelpFormatter):
        pass
    parser = ap.ArgumentParser(description=description,
                               formatter_class=Formatter)
    parser.add_argument("src", help="input root direcotry path")
    parser.add_argument("--min_after_dequeue", type=int, default=1000,
                        help="min_after_dequeue")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="batch size")
    parser.add_argument("--num_threads", type=int, default=1,
                        help="num threads")
    args = parser.parse_args()
    minibatch(args.src, args.min_after_dequeue,
              args.batch_size, args.num_threads)


if __name__ == "__main__":
    main()
