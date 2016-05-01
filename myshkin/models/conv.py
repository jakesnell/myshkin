from attrdict import AttrDict
from collections import namedtuple

import tensorflow as tf
import keras
from keras.layers import Activation, Dense, Dropout, Flatten, Convolution2D, MaxPooling2D

from myshkin.components import *
from myshkin.mixins.model import Model

ConvOpts = namedtuple('ConvOpts', [
    'h', # input height
    'conv_dims', # convolutional dimensions
    'hid_dims', # hidden dimensions
    'c', # output dimension
    'dropout' # dropout probability
])

class Conv(Model):
    def __init__(self, opts, name=None, train=True):
        if name is None:
            name = self.__class__.__name__

        with tf.name_scope(name):
            self.opts = opts

            self.pre_log_classifier = keras.models.Sequential()

            layer_ind = 0
            for next_dim in opts.conv_dims:
                self.pre_log_classifier.add(Convolution2D(next_dim, 5, 5,
                                                      name="conv_layer{:d}".format(layer_ind),
                                                      border_mode='same',
                                                      dim_ordering='th',
                                                      input_shape=(1, self.opts.h // (2**layer_ind), self.opts.h // (2**layer_ind))))
                self.pre_log_classifier.add(Activation('relu'))
                self.pre_log_classifier.add(MaxPooling2D(pool_size=(2, 2), border_mode='same', dim_ordering='th'))
                layer_ind += 1

            self.pre_log_classifier.add(Flatten())

            layer_ind = 0
            for next_dim in opts.hid_dims:
                self.pre_log_classifier.add(Dense(next_dim, activation='relu', name="fc_layer{:d}".format(layer_ind)))
                self.pre_log_classifier.add(Dropout(self.opts.dropout))
                layer_ind += 1

            self.final_layer = Dense(self.opts.c, name="fc_layer{:d}".format(layer_ind))

            if train:
                self.x_bhh = tf.placeholder(tf.float32,
                                            shape=[None, self.opts.h, self.opts.h],
                                            name='x_bhh')
                self.y_b = tf.placeholder(tf.int32,
                                          shape=[None],
                                          name='y_b')
                self.view = self.build(self.x_bhh, self.y_b)

    def build(self, x_bhh, y_b=None):
        x_b1hh = tf.reshape(x_bhh, [-1, 1, self.opts.h, self.opts.h])
        z_bm = self.pre_log_classifier(x_b1hh)
        log_y_hat_bc = self.final_layer(z_bm)
        y_hat_bc = tf.nn.softmax(log_y_hat_bc)

        rval = AttrDict({
            'x_bhh': x_bhh,
            'y_hat_bc': y_hat_bc,
            'z_bm': z_bm
        })

        if y_b is not None:
            loss_b = tf.nn.sparse_softmax_cross_entropy_with_logits(log_y_hat_bc, y_b)
            loss = tf.reduce_mean(loss_b)

            acc_b = tf.equal(tf.cast(tf.argmax(y_hat_bc, 1), tf.int32), y_b)
            acc = tf.reduce_mean(tf.cast(acc_b, tf.float32))
            err = 1.0 - acc

            rval['y_b'] = y_b
            rval['loss'] = loss
            rval['acc'] = acc
            rval['err'] = err

        return rval
