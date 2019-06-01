import numpy as np
import tensorflow as tf
import argparse
import sys
import os
import functools
from collections import defaultdict
import enet

from data_reading import calc_input_tensors
import data_reading
from trainer import Trainer
tf.set_random_seed(42)
np.random.seed(42)
N_CLASSES = 2

def isiterable(obj):
    try:
        tmp = iter(obj)
    except TypeError:
        return False
    return True

def model_assessment_function(metric_vals):
    metric_vals_converted = []
    for elem in metric_vals:
        if isiterable(elem):
            metric_vals_converted.extend(elem)
        else:
            metric_vals_converted.append(elem)
    metric_vals = np.array(metric_vals_converted)
    metric_vals[-N_CLASSES-1:] *= -1
    metric_vals = metric_vals[-N_CLASSES-1:]
    return metric_vals

def is_better_model_func(new, current):
    return np.any(new < current)

def update_model_quality(new, current):
    return np.minimum(new, current)

def load_learning_rate(cli_learning_rate=1e-3, filename=None):
    if filename is None:
        return cli_learning_rate
    with open(filename) as f:
        a = f.readlines()
        return float(a[0].strip())

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def calc_accuracy(predictions, labels, n_classes):
    #return 100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0]
    predictions = tf.cast(tf.argmax(predictions,-1), tf.int32)
    correct_prediction = tf.equal(predictions, labels)
    global_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100
    accuracy_per_class = []
    for i in xrange(n_classes):
        mask = tf.equal(labels, i)
        n_labels = tf.reduce_sum(tf.cast(mask, tf.int32))
        accuracy_per_class.append(
            tf.cond(
                tf.equal(n_labels, 0),
                lambda : tf.constant(100.0),
                lambda : tf.reduce_mean(tf.cast(tf.boolean_mask(correct_prediction, mask), tf.float32))*100))
    mean_class_accuracy = tf.reduce_mean(accuracy_per_class)
    return global_accuracy, mean_class_accuracy, accuracy_per_class

def calc_IoU_metric(predictions, labels, n_classes):
    predictions = tf.cast(tf.argmax(predictions, -1), tf.int32)
    intersection = []
    union = []
    for i in xrange(n_classes):
        a = tf.equal(predictions, i)
        b = tf.equal(labels, i)
        intersection.append(tf.reduce_sum(tf.cast(tf.logical_and(a,b), tf.float32)))
        union.append(tf.reduce_sum(tf.cast(tf.logical_or(a,b), tf.float32)))
    return intersection, union

def aggregate_IoU_results(data, n_classes=N_CLASSES, *args, **kwargs):
    intersection = np.zeros(n_classes, np.float32)
    union = np.zeros(n_classes, np.float32)
    for batch_intersection, batch_union in data:
        intersection += np.array(batch_intersection)
        union += np.array(batch_union)
    iou = np.where(union == 0, 1, intersection / union)
    return [('mean iou', np.mean(iou)), ("class iou", iou)]


def create_graph(input_data, labels, is_training, n_classes=N_CLASSES):
    logits = enet.build_graph2(input_data, is_training, n_classes)
    print logits.get_shape()
    label_to_weight = tf.constant(np.array([0.3, 1.], np.float32))
    weights = tf.gather(label_to_weight, labels)  # FIXME
    loss = tf.reduce_mean(weights * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    predictions = tf.nn.softmax(logits)
    return predictions, labels, loss


def run_training(args):
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            train_data = calc_input_tensors(
                    args.train_data, batch_size=args.batch_size,
                    name_scope='train_input', do_shuffle=True)
            validation_data = calc_input_tensors(
                    args.validation_data,
                    batch_size=1,
                    name_scope='val_input', do_shuffle=False)

        # 2. build graph
        is_training = tf.placeholder(tf.bool, name="is_training")
        learning_rate_ph = tf.placeholder(tf.float32, name="learning_rate")
        global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)

        graph_template = tf.make_template('Graph', create_graph,
                n_classes=N_CLASSES)


        train_batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
                train_data, capacity=2*args.n_gpus)
        train_losses = []
        train_metrics = defaultdict(list)
        tower_grads = []

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_ph)

        for i in xrange(args.n_gpus):
            train_data = train_batch_queue.dequeue()
            with tf.device("/gpu:{}".format(i)), tf.name_scope('train_tower_{}'.format(i)):
                predictions, labels, train_loss = graph_template(*train_data, is_training=True)
                glob_accuracy, mean_accuracy, accuracy_per_class = calc_accuracy(
                                                predictions, labels, N_CLASSES)
                train_metrics['global_accuracy'].append(glob_accuracy)
                train_metrics['mean_accuracy'].append(mean_accuracy)
                train_metrics['accuracy_per_class'].append(accuracy_per_class)
                train_losses.append(train_loss)
                tower_grads.append(optimizer.compute_gradients(train_loss))

        val_batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
                validation_data, capacity=2*args.n_gpus)
        val_metrics = defaultdict(list)
        iou_stat = []
        for i in xrange(args.n_gpus):
            val_data = val_batch_queue.dequeue()
            with tf.device("/gpu:{}".format(i)), tf.name_scope('val_tower_{}'.format(i)):
                predictions, labels, val_loss = graph_template(*val_data, is_training=False)
                glob_accuracy, mean_accuracy, accuracy_per_class = calc_accuracy(
                                                predictions, labels, N_CLASSES)
                val_metrics['global_accuracy'].append(glob_accuracy)
                val_metrics['mean_accuracy'].append(mean_accuracy)
                val_metrics['accuracy_per_class'].append(accuracy_per_class)
                val_metrics['target_loss'].append(val_loss)
                intersection, union = calc_IoU_metric(predictions, labels, N_CLASSES)
                iou_stat.append([intersection, union])

        with tf.device('/gpu:0'):
        #with tf.device('/cpu:0'):
            train_loss = tf.div(tf.add_n(train_losses), args.n_gpus)
            train_metrics_averaged = [("target_loss", train_loss)]
            for k, v in train_metrics.iteritems():
                train_metrics_averaged.append((k, tf.div(tf.add_n(v), args.n_gpus)))
            val_metrics_averaged = []
            for k, v in val_metrics.iteritems():
                val_metrics_averaged.append((k, tf.div(tf.add_n(v), args.n_gpus)))
            iou_stat = (tf.add_n([elem[0] for elem in iou_stat]),
                         tf.add_n([elem[1] for elem in iou_stat]))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grads = average_gradients(tower_grads)
            glob_norm = tf.global_norm(grads)
            optimizer = optimizer.apply_gradients(grads, global_step=global_step)

        train_metrics_averaged.append(("glob_norm", glob_norm))
        args.validation_frequency = int(args.validation_frequency / args.batch_size / args.n_gpus)
        learning_rate_value = load_learning_rate(args.learning_rate, args.learning_rate_file)

        print "Learning rate: ", learning_rate_value

        with Trainer(optimizer, train_metrics_averaged, val_metrics_averaged, args.batch_size,
                args.epoch_size, args.output_models_folder,
                val_tensors_for_accumulation=iou_stat,
                placeholders=[learning_rate_ph],
                is_training=is_training, n_gpus=args.n_gpus) as trainer:
            if args.checkpoint_file:
                trainer.restore(args.checkpoint_file)

            while trainer.ok():
                trainer.do_train_step([learning_rate_value], True)

                if trainer.step % args.stat_frequency == 0:
                    trainer.print_local_stat()
                    trainer.reset_local_stat()

                if trainer.step % args.validation_frequency == 0:
                    trainer.print_global_stat()
                    trainer.reset_global_stat()
                    trainer.validate(args.validation_size, model_assessment_function,
                            is_better_model_func, update_model_quality, print_stat=True,
                            placeholders_vals=[learning_rate_value],
                            accumulated_tensors_processing_func=aggregate_IoU_results)
                    if trainer.need_to_save or trainer.not_to_saved > 10:
                        trainer.save()
                        trainer.not_to_saved = 0
                    else:
                        trainer.not_to_saved += 1
                    new_lr = load_learning_rate(args.learning_rate, args.learning_rate_file)
                    if new_lr != learning_rate_value:
                        print "Learning rate was changed: {} -> {}".format(learning_rate_value, new_lr)
                        learning_rate_value = new_lr


def main(args):
    run_training(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TBD')
    parser.add_argument("-t", "--train-data", dest="train_data", required=True, type=str, nargs='+',
            metavar="FILE", help='train dataset as tfrecords file')
    parser.add_argument("-v", "--validation-data", dest="validation_data", required=True, type=str, nargs="+",
            metavar="FILE", help='validation dataset as tfrecords file')
    parser.add_argument("-lr", "--learning-rate", dest="learning_rate", default=0.001, type=str,
            metavar="FLOAT", help='learning rate')
    parser.add_argument("-lrf", "--learning-rate-file", dest="learning_rate_file", default=None, type=str,
            metavar="FILE", help='learning rate file (overrides value for learning rate)')
    parser.add_argument("-b", "--batch-size", dest="batch_size", required=True, type=int,
            metavar="INT", help='batch size')
    parser.add_argument("-g", "--n_gpus", dest="n_gpus", default=1, type=int,
            metavar="INT", help='number of gpus')
    parser.add_argument("-n", "--num-epochs", dest="num_epochs", required=True, type=int,
            metavar="INT", help='number of epochs')
    parser.add_argument("-m", "--models", dest="output_models_folder", required=True, type=str,
            metavar="FOLDER", help="output folder where models will be saved")

    parser.add_argument("-sf", "--stat-frequency", dest="stat_frequency", default=20, type=int,
            metavar="INT", help="how often train error statistic will be printed (amount of batches)")
    parser.add_argument("-vf", "--validation-frequency", dest="validation_frequency", required=True, type=int,
            metavar="INT", help="how often validation error will be computed (amount of samples)")
    parser.add_argument("-vs", "--validation-size", dest="validation_size", required=True, type=int,
            metavar="INT", help="amount of batches in validation phase")
    parser.add_argument("-e", "--epoch-size", dest="epoch_size", required=True, type=int,
            metavar="INT", help="amount of samples (!) in a epoch")
    parser.add_argument("-c", "--checkpoint", dest="checkpoint_file", default=None, type=str,
            metavar="FILE", help="(optional) checkpoint to continue the training")

    args = parser.parse_args()

    print "Args: ", args

    main(args)

