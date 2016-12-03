#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import argparse as ap
import numpy as np
import tensorflow as tf


def _int64_feature(value):
    assert(type(value) == int)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_list_feature(value):
    if type(value) == np.ndarray:
        value = value.flatten().tolist()
    assert(type(value) == list)
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def write_tfrecords(dst_path, num_files, num_records):
    for fidx in range(num_files):
        file_path = os.path.join(dst_path, "file{0:02d}.tfrecord".format(fidx))
        with tf.python_io.TFRecordWriter(file_path) as writer:
            print "wirte: {0}".format(file_path)
            for ridx in range(num_records):
                data = np.random.rand(2, 4)
                example = tf.train.Example(features=tf.train.Features(feature={
                    'file_idx': _int64_feature(fidx),
                    'record_idx': _int64_feature(ridx),
                    'data': _float_list_feature(data)
                }))
                writer.write(example.SerializeToString())


def main():
    description = """write TFRecord files for test

[dst]
├── file01.tfrecord
├── ...
└── file09.tfrecord
"""

    class Formatter(ap.ArgumentDefaultsHelpFormatter,
                    ap.RawDescriptionHelpFormatter):
        pass
    parser = ap.ArgumentParser(description=description,
                               formatter_class=Formatter)
    parser.add_argument("dst", help="output root direcotory path")
    parser.add_argument("--num_files", type=int, default=10,
                        help="number of TFRecord files")
    parser.add_argument("--num_records", type=int, default=1000,
                        help="number of records in a TFRecords")
    args = parser.parse_args()
    write_tfrecords(args.dst, args.num_files, args.num_records)


if __name__ == "__main__":
    main()
