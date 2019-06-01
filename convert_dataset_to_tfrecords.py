"""Converts MNIST data to TFRecords file format with Example protos."""
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import os
from PIL import Image
import numpy as np
import tensorflow as tf
import argparse
import scipy as sp
import scipy.misc
import random
#from tensorflow.examples.tutorials.mnist import input_data
import json
import time
import skimage.morphology
import tqdm

import cv2

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(input_folder, output_filename_train, output_filename_val):
    input_names = []
    input_names = os.listdir(input_folder)
    random.shuffle(input_names)
    input_names = [os.path.join(input_folder, elem) for elem in input_names]

    num_examples = len(input_names)

    writer_train = tf.python_io.TFRecordWriter(output_filename_train)
    writer_val = tf.python_io.TFRecordWriter(output_filename_val)
    n_skipped = 0
    n_train, n_test = 0, 0
    print "Total: ", num_examples
    n_positive = 0
    n_negative = 0
    for dir_name in tqdm.tqdm(input_names):
        mask_file = os.path.join(dir_name, '17', 'buildings.png')
        image_file = os.path.join(dir_name, '17', 'satellite.png')
        if not os.path.exists(mask_file) or not os.path.exists(image_file):
            n_skipped += 1
            continue
        mask = sp.misc.imread(mask_file)[:,:,3]
        house_mask = mask > 0
        if not np.any(house_mask):
            n_skipped += 1
            continue
        n_positive += np.sum(house_mask)
        n_negative += np.sum(mask == 0)
        with open(image_file) as f:
            image_data = f.read()

        with open(mask_file) as f:
            mask_data = f.read()

        example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': _bytes_feature(image_data),
                'label_raw': _bytes_feature(mask_data)}))
        if random.random() > 0.1:
            writer_train.write(example.SerializeToString())
            n_train += 1
        else:
            writer_val.write(example.SerializeToString())
            n_test += 1
    writer_train.close()
    writer_val.close()

    print "n_train: {} ; n_test: {} ; n_skipped: {}".format(n_train, n_test, n_skipped)
    print "n_positive: {}; n_negative: {}".format(n_positive, n_negative)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TBD')
    parser.add_argument("-i", "--input", dest="input_folder", required=True, type=str,
            metavar="FOLDER", help='input folder with imgs')
    #parser.add_argument("-m", "--masks", dest="masks_folder", required=True, type=str, nargs="+",
    #        metavar="FOLDER", help='input folder with masks')
    parser.add_argument("-ot", "--output-train", dest="output_file_train", required=True, type=str,
            metavar="FILE", help='output file with tfrecords')
    parser.add_argument("-ov", "--output-val", dest="output_file_val", required=True, type=str,
            metavar="FILE", help='output file with tfrecords')

    args = parser.parse_args()

    convert_to(args.input_folder, args.output_file_train, args.output_file_val)


