import datetime
from operator import add
import os
import time

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from PIL import Image

from myshkin.util.feeder import reduce_batches
from myshkin.util.vis import patch_view, interleave

class Callback(object):
    def get_monitor_fields(self):
        raise NotImplementedError("get_monitor_fields not implemented for {:s}".format(self))

    def stop_training(self):
        return False

    def register_sess(self, sess):
        pass

    def stop_training_message(self):
        raise NotImplementedError()

class DefaultLogger(Callback):
    def __init__(self, monitor_fields):
        self.monitor_fields = monitor_fields

        self.t_start = time.time()

    def get_monitor_fields(self):
        return self.monitor_fields

    def epoch_end(self, epoch_ind, sess, model, train_stats, valid_stats):
        t_next = time.time()
        t_elapsed = t_next - self.t_start
        self.t_start = t_next

        train_strings = []
        valid_strings = []

        for field in self.monitor_fields:
            train_strings.append("train {:s} = {:0.8f}".format(field, train_stats[field]))
            valid_strings.append("valid {:s} = {:0.8f}".format(field, valid_stats[field]))

        print "Epoch {:d}: {:s}, {:s} ({:0.2f}s)".format(epoch_ind, ", ".join(train_strings), ", ".join(valid_strings), t_elapsed)

class DeepDashboardLogger(Callback):
    def __init__(self, exp_id, loggers):
        self.exp_id = exp_id
        self.loggers = loggers

        webserver_root = os.getenv('APACHE_ROOT')
        if webserver_root is None:
            self.outdir = exp_id
        else:
            self.outdir = os.path.join(webserver_root, 'results', exp_id)

        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)

        self.write_catalog(os.path.join(self.outdir, 'catalog'))

        for logger in self.loggers:
            logger.register_master(self)

    def get_monitor_fields(self):
        return list(set(reduce(add, [logger.get_monitor_fields() for logger in self.loggers])))

    def write_catalog(self, file_name):
        with open(file_name, 'w') as fid:
            fid.write("filename,type,name\n")

            for logger in self.loggers:
                fid.write(logger.get_catalog_entry()+"\n")

    def epoch_end(self, epoch_ind, sess, model, train_stats, valid_stats):
        for logger in self.loggers:
            logger.epoch_end(epoch_ind, sess, model, train_stats, valid_stats)

class DeepDashboardSubLogger(Callback):
    def register_master(self, master):
        self.master = master

    def get_catalog_entry(self):
        raise NotImplementedError()

    def get_file_name(self):
        return os.path.join(self.master.outdir, self.base_file)

class RawLogger(DeepDashboardSubLogger):
    def __init__(self, base_file, label, monitor_fields):
        self.base_file = base_file
        self.label = label
        self.monitor_fields = monitor_fields

        self.t_start = time.time()

    def get_catalog_entry(self):
        return "{:s},plain,{:s}".format(os.path.basename(self.base_file), self.label)

    def get_monitor_fields(self):
        return self.monitor_fields

    def register_master(self, master):
        DeepDashboardSubLogger.register_master(self, master)

        with open(self.get_file_name(), 'w') as fid:
            pass

    def epoch_end(self, epoch_ind, sess, model, train_stats, valid_stats):
        t_next = time.time()
        t_elapsed = t_next - self.t_start
        self.t_start = t_next

        train_strings = []
        valid_strings = []

        for field in self.monitor_fields:
            train_strings.append("train {:s} = {:0.8f}".format(field, train_stats[field]))
            valid_strings.append("valid {:s} = {:0.8f}".format(field, valid_stats[field]))

        with open(self.get_file_name(), 'a') as fid:
            fid.write("Epoch {:d}: {:s}, {:s} ({:0.2f}s)\n".format(epoch_ind, ", ".join(train_strings), ", ".join(valid_strings), t_elapsed))

class CSVLogger(DeepDashboardSubLogger):
    def __init__(self, base_file, label, monitor_fields):
        self.base_file = base_file
        self.label = label
        self.monitor_fields = monitor_fields
        self.train_labels = ["train {:s}".format(field) for field in monitor_fields]
        self.valid_labels = ["valid {:s}".format(field) for field in monitor_fields]

    def get_catalog_entry(self):
        return "{:s},csv,{:s}".format(os.path.basename(self.base_file), self.label)

    def get_monitor_fields(self):
        return self.monitor_fields

    def register_master(self, master):
        DeepDashboardSubLogger.register_master(self, master)

        with open(self.get_file_name(), 'w') as fid:
            fid.write("step,time,{:s},{:s}\n".format(",".join(self.train_labels),
                                                     ",".join(self.valid_labels)))

    def epoch_end(self, epoch_ind, sess, model, train_stats, valid_stats):
        time_str = datetime.datetime.utcnow().isoformat()

        with open(self.get_file_name(), 'a') as fid:
            train_strings = ["{:0.8f}".format(train_stats[field]) for field in self.monitor_fields]
            valid_strings = ["{:0.8f}".format(valid_stats[field]) for field in self.monitor_fields]
            fid.write("{:d},{:s},{:s},{:s}\n".format(epoch_ind,
                                                     time_str,
                                                     ",".join(train_strings),
                                                     ",".join(valid_strings)))

class ImageLogger(DeepDashboardSubLogger):
    def __init__(self, base_file, label, image_tensors, feeder, n_cols):
        self.base_file = base_file
        self.label = label
        self.image_tensors = image_tensors
        self.feeder = feeder
        self.n_cols = n_cols

    def get_catalog_entry(self):
        return "{:s},image,{:s}".format(os.path.basename(self.base_file), self.label)

    def get_monitor_fields(self):
        return []

    def epoch_end(self, epoch_ind, sess, model, train_stats, valid_stats):
        images = reduce_batches(sess,
                    {
                        i: image_tensor
                        for (i, image_tensor) in enumerate(self.image_tensors)
                    },
                    self.feeder,
                    shuffle=False)
        image_arr = interleave([images[i] for i in xrange(len(self.image_tensors))])
        plt.imsave(self.get_file_name(), patch_view(image_arr, self.n_cols))

class EarlyStopping(Callback):
    def __init__(self, monitor_field, patience):
        self.monitor_field = monitor_field
        self.patience = patience

        self.opt_val = np.infty
        self.opt_epoch_ind = -1

        self._stop_training = False

    def get_monitor_fields(self):
        return [self.monitor_field]

    def epoch_end(self, epoch_ind, sess, model, train_stats, valid_stats):
        cur_val = valid_stats[self.monitor_field]

        if cur_val < self.opt_val:
            self.opt_val = cur_val
            self.opt_epoch_ind = epoch_ind
            self._stop_training = False
        else:
            if epoch_ind - self.opt_epoch_ind > self.patience:
                self._stop_training = True

    def stop_training(self):
        return self._stop_training

    def stop_training_message(self):
        return "Patience {:d} exceeded: minimum validation {:s} of {:0.8f} achieved at epoch {:d}".format(self.patience, self.monitor_field, self.opt_val, self.opt_epoch_ind)

class ModelCheckpoint(Callback):
    def __init__(self, out_dir, monitor_field, verbose=False):
        self.out_dir = out_dir
        self.monitor_field = monitor_field
        self.verbose = verbose

        self.opt_val = np.infty
        self.saver = tf.train.Saver()

        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

    def get_monitor_fields(self):
        return [self.monitor_field]

    def epoch_end(self, epoch_ind, sess, model, train_stats, valid_stats):
        cur_val = valid_stats[self.monitor_field]

        if cur_val < self.opt_val:
            model_out_file = os.path.join(self.out_dir, "model.ckpt".format(epoch_ind))
            conf_out_file = os.path.join(self.out_dir, 'model_conf.yaml')

            self.opt_val = cur_val
            t_start = time.time()
            self.saver.save(self.sess, model_out_file)
            model.save_conf(conf_out_file)
            t_elapsed = time.time() - t_start

            if self.verbose:
                print "> validation {:s} of {:0.8f}, saved to {:s} ({:0.2f}s)".format(self.monitor_field, cur_val, model_out_file, t_elapsed)

    def register_sess(self, sess):
        self.sess = sess
