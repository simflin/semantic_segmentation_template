import numpy as np
import tqdm
import scipy as sp
import scipy.misc
import tensorflow as tf
import argparse
import sys
import os
import enet
import unet
import random

import skimage.color
import skimage
N_CLASSES=2


class Net(object):
    def __init__(self, network_state_filename, network_type, is_training=False, n_classes=N_CLASSES):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope("Graph"):
                self.graph_input = tf.placeholder(tf.float32, (None, None, None, 3), name="graph_input")
                self.is_training = tf.constant(is_training, dtype=tf.bool)
                n_input_channels = 3
                if network_type == 'unet_wo_psp':
                    logits = unet.build_graph2(self.graph_input, self.is_training, n_classes)
                else:
                    assert False, network_type

                self.batch_predictions = tf.nn.softmax(logits)
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                NUM_THREADS = 2
                config.intra_op_parallelism_threads=NUM_THREADS
                config.inter_op_parallelism_threads=NUM_THREADS
                saver = tf.train.Saver()
                self.session = tf.Session(config=config)
                saver.restore(self.session, network_state_filename)

    def segment_image(self, img):
        height, width = img.shape[:2]
        img = img[:,:,:3]

        img = img[np.newaxis]  # batch size dimension
        img = img.astype(np.float32) / 255 - 0.5

        [predictions] = self.session.run([self.batch_predictions],
                feed_dict={self.graph_input: img})
        n_classes = predictions.shape[-1]
        predictions = predictions.reshape(height, width, n_classes)

        return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TBD')
    parser.add_argument("-i", "--input", dest="input_file_name", required=True, type=str,
            metavar="FILE", help='input image')
    parser.add_argument("-o", "--output", dest="output_file_name_prefix", required=True, type=str,
            metavar="FILE", help='output file name prefix')
    parser.add_argument("-s", "--state", dest="network_state_file_name", required=True, type=str,
            metavar="FILE", help="file with network state")
    parser.add_argument("-t", "--type", dest="type", default='unet_wo_psp', type=str,
            metavar="FILE", help="network type")

    args = parser.parse_args()

    if not os.path.exists(args.output_file_name_prefix):
        os.mkdir(args.output_file_name_prefix)

    enet = Net(args.network_state_file_name, args.type)
    input_names = os.listdir(args.input_file_name)
    random.shuffle(input_names)
    input_names = [os.path.join(args.input_file_name, elem) for elem in input_names]
    i = 0

    for dir_name in tqdm.tqdm(input_names):
        mask_file = os.path.join(dir_name, '17', 'buildings.png')
        image_file = os.path.join(dir_name, '17', 'satellite.png')
        if not os.path.exists(mask_file) or not os.path.exists(image_file):
            continue
        mask = sp.misc.imread(mask_file)[:,:,3]
        house_mask = mask > 0
        if not np.any(house_mask):
            continue
        im = sp.misc.imread(image_file)[:,:,:3]
        out = enet.segment_image(im)
        out = out.argmax(axis=-1)[:,:,None]
        blending = 0.3
        a = np.zeros((512, 30, 3), np.uint8)
        color = np.array([255,0,0], np.float32).reshape(1,1,3)
        bl1 = (im * (1. - blending) + out * color * blending).astype(np.uint8)
        bl2 = (im * (1. - blending) + house_mask.astype(np.float32)[:,:,None]*color * blending).astype(np.uint8)
        im = np.hstack([bl1,a, bl2])

        sp.misc.imsave(os.path.join(args.output_file_name_prefix, "{}.png".format(i)), im)
        i += 1
        if i > 200:
            break
