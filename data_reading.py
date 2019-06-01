import tensorflow as tf
import numpy as np


def read_and_decode(filename_queue, use_random_crop=True):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example,
                               features={
                                  'image_raw': tf.FixedLenFeature([],tf.string),
                                  'label_raw': tf.FixedLenFeature([],tf.string),
                               })
    image = tf.image.decode_png(features['image_raw'], channels=3, dtype=tf.uint8, name="decode_sat")
    label = tf.image.decode_png(features['label_raw'], channels=4, dtype=tf.uint8, name="decode_mask")
    label = label[:,:,3] > 0
    label = tf.cast(label, tf.uint8)

    if use_random_crop:
        # augmentation
        data = tf.concat([image, label[:,:,tf.newaxis]], axis=-1)
        data = tf.image.random_flip_left_right(data)
        data = tf.image.random_flip_up_down(data)

        offset_height = tf.random_uniform([], minval=0, maxval=512-256, dtype=tf.int32)
        offset_width = tf.random_uniform([], minval=0, maxval=512-256, dtype=tf.int32)
        data = data[offset_height: offset_height+256, offset_width: offset_width+256]
        image = data[:,:,:3]
        label = data[:,:,3]
        label.set_shape([256,256])
        image.set_shape([256,256,3])
    else:
        label.set_shape([512,512])
        image.set_shape([512,512,3])

    image = tf.cast(image, tf.float32) * (1. / 255.) - 0.5
    label = tf.cast(label, tf.int32)

    return image, label


def calc_input_tensors(filenames, batch_size, num_epochs=None, name_scope='input',
                       do_shuffle=False):
    with tf.name_scope(name_scope):
        filename_queue = tf.train.string_input_producer(
                filenames, num_epochs=num_epochs)

        sample_data = read_and_decode(filename_queue, do_shuffle)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        if do_shuffle:
            batch_data = tf.train.shuffle_batch(
                    sample_data, batch_size=batch_size, num_threads=24,
                    #capacity=1000 + 3 * batch_size,
                    # Ensures a minimum amount of shuffling of examples.
                    #min_after_dequeue=1000)
                    capacity=100 + 3 * batch_size,
                    min_after_dequeue=100)
        else:
            batch_data = tf.train.batch(
                    sample_data, batch_size=batch_size, num_threads=24,
                    capacity=100 + 3 * batch_size)

        return batch_data

