#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse as ap
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib import cm
import random


def show_batch_bias(batch_contents, num_files, batch_size, sampling_rate=0.05):
    num_records = len(batch_contents) * batch_size / num_files
    data = defaultdict(list)
    for batch_idx, batch in enumerate(batch_contents):
        x = batch_idx
        for file_idx, record_idx in batch:
            if random.random() >= sampling_rate:
                continue
            y = record_idx
            data[file_idx].append((x, y))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for file_idx, xy in data.items():
        x, y = zip(*xy)
        color = cm.jet(1.0 * file_idx / num_files)
        ax.scatter(x, y, c=color, label="file{0:02d}".format(file_idx))

    ax.set_title('second scatter plot')
    ax.set_xlabel('batch')
    ax.set_ylabel('record')

    ax.legend(loc='lower right')
    ax.set_xlim([0, len(batch_contents)])
    ax.set_ylim([0, num_records])

    plt.show()


def read_feature_file(filename_queue):
    reader = tf.TFRecordReader()
    key, selialized_data = reader.read(filename_queue)
    features_def = {
        "file_idx": tf.FixedLenFeature([1], tf.int64),
        "record_idx": tf.FixedLenFeature([1], tf.int64),
        "data": tf.FixedLenFeature([2, 4], tf.float32)
    }
    example = tf.parse_single_example(selialized_data,
                                      features=features_def)
    return example


def minibatch(src_path, board_logdir,
              min_after_dequeue, batch_size, num_threads, join_flag):
    filepaths = glob.glob("{src}/*.tfrecord".format(src=src_path))
    filename_queue = tf.train.string_input_producer(filepaths,
                                                    num_epochs=1,
                                                    shuffle=True)
    capacity = min_after_dequeue + 3 * batch_size

    if join_flag:
        example_list = [read_feature_file(filename_queue)
                        for _ in range(num_threads)]
        features_batch = tf.train.shuffle_batch_join(
            example_list,
            batch_size=batch_size,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            allow_smaller_final_batch=True
        )
    else:
        features_batch = tf.train.shuffle_batch(
            read_feature_file(filename_queue),
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
        merge_summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(board_logdir, sess.graph)
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            iter = 0
            while not coord.should_stop():
                sample = sess.run(features_batch)
                # optimization or evaluation with sample["data"]
                contents = zip(sample["file_idx"].flatten(),
                               sample["record_idx"].flatten())
                batch_contents.append(contents)
                # for tensorboard
                summary = sess.run(merge_summary_op)
                summary_writer.add_summary(summary, iter)
                iter += 1
        except tf.errors.OutOfRangeError:
            pass
        finally:
            coord.request_stop()
            coord.join(threads)

    show_batch_bias(batch_contents, len(filepaths), batch_size)


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
    parser.add_argument("logdir", help="tensorboard log direcotry path")
    parser.add_argument("--min_after_dequeue", type=int, default=10000,
                        help="min_after_dequeue")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="batch size")
    parser.add_argument("--num_threads", type=int, default=1,
                        help="num threads")
    parser.add_argument("--join", default=False,
                        action="store_true", help="use shuffle_batch_join")
    args = parser.parse_args()
    minibatch(args.src, args.logdir, args.min_after_dequeue,
              args.batch_size, args.num_threads, args.join)


if __name__ == "__main__":
    main()
