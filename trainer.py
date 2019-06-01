import os
import tensorflow as tf
import numpy as np
import time
import sys


class Trainer(object):
    def __init__(self, optimizer, train_metrics, val_metrics, batch_size, epoch_size, output_models_folder,
                    val_tensors_for_accumulation=[],
                    placeholders=None, is_training=None, n_gpus=1):
        self.optimizer = optimizer
        assert isinstance(train_metrics, list), type(train_metrics)
        assert isinstance(val_metrics, list), type(val_metrics)
        self.train_metrics = [elem[1] for elem in train_metrics]
        self.train_metric_names = [elem[0] for elem in train_metrics]
        self.val_metrics = [elem[1] for elem in val_metrics]
        self.val_metric_names = [elem[0] for elem in val_metrics]
        self.val_tensors_for_accumulation = val_tensors_for_accumulation
        self.placeholders = placeholders
        self.step = 0

        self.local_stat = None
        self.global_stat = None

        self.local_stat_init_step = 0
        self.global_stat_init_step = 0
        self.local_stat_start_time = 0
        self.global_stat_start_time = 0

        self.batch_size = batch_size
        self.n_gpus = n_gpus
        self.epoch_size = epoch_size
        self.output_models_folder = output_models_folder
        self.prev_model_quality = None
        self.need_to_save = False
        self.is_training = is_training
        if not os.path.exists(output_models_folder):
            os.mkdir(output_models_folder)

        self.not_to_saved = 0


    def __enter__(self):
        init_op = tf.initialize_all_variables()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement=True
        NUM_THREADS = 16
        config.intra_op_parallelism_threads=NUM_THREADS
        config.inter_op_parallelism_threads=NUM_THREADS
        self.saver = tf.train.Saver(max_to_keep=None)
        self.session = tf.Session(config=config)
        self.session.__enter__()
        self.session.run(init_op)
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.session, coord=self.coord)
        return self

    def __exit__(self, exec_type, exec_value, exec_tb):
        self.save(name="epoch_final.ckpt")
        self.coord.request_stop()
        self.coord.join(self.threads)
        self.session.__exit__(exec_type, exec_value, exec_tb)

    def restore(self, filename):
        self.saver.restore(self.session, filename)
        self.step = tf.train.global_step(self.session, tf.train.get_global_step())

    def fetch(self, tensors, feed_dict=None):
        return self.session.run(tensors, feed_dict=feed_dict)

    def __create_feed_dict(self, placeholders_vals=None, train=False):
        assert (placeholders_vals is None) == (self.placeholders is None)
        if placeholders_vals is not None:
            assert len(placeholders_vals) == len(self.placeholders)
            feed_dict = dict(zip(self.placeholders, placeholders_vals))
        else:
            feed_dict = None
        if self.is_training is not None:
            if feed_dict is None:
                feed_dict = {}
            feed_dict[self.is_training] = train
        return feed_dict

    create_feed_dict = __create_feed_dict

    def do_train_step(self, placeholders_vals=None, is_training_val=True):
        start_time = time.time()
        metrics_vals = self.fetch([self.optimizer] + self.train_metrics,
                feed_dict=self.__create_feed_dict(placeholders_vals, train=is_training_val))[1:]
        if self.local_stat is None:
            self.local_stat = [np.array(elem) for elem in metrics_vals]
            self.local_stat_init_step = self.step
            self.local_stat_start_time = start_time
        else:
            for i in xrange(len(self.local_stat)):
                self.local_stat[i] += np.array(metrics_vals[i])

        if self.global_stat is None:
            self.global_stat = [np.array(elem) for elem in metrics_vals]
            self.global_stat_init_step = self.step
            self.global_stat_start_time = start_time
        else:
            for i in xrange(len(self.global_stat)):
                self.global_stat[i] += np.array(metrics_vals[i])
        self.step += 1

    def validate(self, val_size, model_assessment_func, is_better_model_func, update_model_quality,
            accumulated_tensors_processing_func=None,
            print_stat=False,placeholders_vals=None):
        results = None
        accumulated_tensors = []
        #n_steps = val_size / self.batch_size / self.n_gpus
        n_steps = val_size / self.n_gpus
        for i in xrange(n_steps):
            metrics_val, batch_accumulated_tensors = self.fetch(
                    [self.val_metrics, self.val_tensors_for_accumulation],
                    feed_dict=self.__create_feed_dict(placeholders_vals, train=False))
            accumulated_tensors.append(batch_accumulated_tensors)
            if results is None:
                results = metrics_val
            else:
                for i in xrange(len(metrics_val)):
                    results[i] += np.array(metrics_val[i])
        results = [elem/ n_steps for elem in results]
        if accumulated_tensors_processing_func is not None:
            accum_results_metrics = accumulated_tensors_processing_func(accumulated_tensors)
            accum_metrics_names = [elem[0] for elem in accum_results_metrics]
            accum_metrics_vals = [elem[1] for elem in accum_results_metrics]
            results.extend(accum_metrics_vals)
        else:
            accum_metrics_names = []
        model_quality = model_assessment_func(results)

        if self.prev_model_quality is None or is_better_model_func(model_quality, self.prev_model_quality):
            if self.prev_model_quality is None:
                self.prev_model_quality = model_quality
            else:
                self.prev_model_quality = update_model_quality(model_quality, self.prev_model_quality)
            self.need_to_save = True

        if print_stat:
            msg = "Validation ({:.2f} epoch): quality: {}; best quality {}".format(
                    float(self.step) * self.batch_size * self.n_gpus / self.epoch_size, model_quality.tolist(), self.prev_model_quality.tolist())
            msg += ";".join([" {}: {}".format(m, v.tolist()) for m, v in zip(self.val_metric_names + accum_metrics_names, results)])
            print msg
            sys.stdout.flush()

    def print_local_stat(self):
        elapsed_time = time.time() - self.local_stat_start_time
        elapsed_batches = self.step - self.local_stat_init_step
        msg = "Step {} (epoch {:.2f}) (batch ts: {:.3f} ms): ".format(
                self.step, float(self.step) * self.batch_size * self.n_gpus / self.epoch_size,
                elapsed_time / elapsed_batches * 1000.)
        stat = [elem / elapsed_batches for elem in self.local_stat]
        msg += ";".join([" {}: {}".format(m, v.tolist()) for m, v in zip(self.train_metric_names, stat)])
        print msg
        sys.stdout.flush()

    def print_global_stat(self):
        elapsed_time = time.time() - self.global_stat_start_time
        elapsed_batches = self.step - self.global_stat_init_step
        msg = "Average train stat (epoch {:.2f}) (total time: {:.3f} sec ".format(
                float(self.step) * self.batch_size * self.n_gpus / self.epoch_size,
                elapsed_time)
        stat = [elem / elapsed_batches for elem in self.global_stat]
        msg += ";".join([" {}: {}".format(m, v.tolist()) for m, v in zip(self.train_metric_names, stat)])
        print msg
        sys.stdout.flush()

    def reset_local_stat(self):
        self.local_stat = None

    def reset_global_stat(self):
        self.global_stat = None

    def save(self, name=None):
        if name is None:
            if self.need_to_save:
                name = "epoch_{:.3f}.ckpt".format(self.step * self.batch_size * self.n_gpus * 1. / self.epoch_size)
            else:
                name = "epoch_{:.3f}-interm.ckpt".format(self.step * self.batch_size * self.n_gpus * 1. / self.epoch_size)
        save_path = self.saver.save(self.session,
                os.path.join(self.output_models_folder, name), global_step=self.step)
        if self.need_to_save:
            print "Model has been truly saved in ", save_path
        else:
            print "Model has been just saved in ", save_path

        self.need_to_save = False

    def ok(self):
        return not self.coord.should_stop()

