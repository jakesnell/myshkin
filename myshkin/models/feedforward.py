from attrdict import AttrDict
from collections import namedtuple

import tensorflow as tf
from keras.layers import Dense, Dropout

from myshkin.components import *
from myshkin.mixins.model import Model

FeedforwardOpts = namedtuple('FeedforwardOpts', [
    'k', # input dimension
    'hid_dims', # hidden dimensions
    'c', # output dimension
    'dropout' # dropout probability
])

class Feedforward(Model):
    def __init__(self, opts):
        self.opts = opts

        self.x_bk = tf.placeholder(tf.float32,
                                   shape=[None, self.opts.k],
                                   name='x_bk')
        self.y_b = tf.placeholder(tf.int32,
                                  shape=[None],
                                  name='y_b')

        layers = []
        cur_dim = self.opts.k
        layer_ind = 0
        for next_dim in opts.hid_dims:
            layers.append(Dense(next_dim, activation='relu', name="layer{:d}".format(layer_ind)))
            layers.append(Dropout(self.opts.dropout))
            cur_dim = next_dim
            layer_ind += 1

        layers.append(Dense(self.opts.c, name="layer{:d}".format(layer_ind)))

        self.log_classifier = Sequential(layers)

        self.view = self.build()

    def build(self):
        log_classifier_seq = self.log_classifier.apply_seq(self.x_bk)
        log_y_hat_bc = log_classifier_seq[-1]
        loss_b = tf.nn.sparse_softmax_cross_entropy_with_logits(log_y_hat_bc, self.y_b)
        loss = tf.reduce_mean(loss_b)

        y_hat_bc = tf.nn.softmax(log_y_hat_bc)

        acc_b = tf.cast(tf.equal(tf.cast(tf.argmax(y_hat_bc, 1), tf.int32), self.y_b), tf.float32)
        err_b = 1.0 - acc_b

        return AttrDict({
            'x_bk': self.x_bk,
            'y_b': self.y_b,
            'y_hat_bc': y_hat_bc,
            'z_bh': log_classifier_seq[-2],
            'loss_b': loss_b,
            'acc_b': acc_b,
            'err_b': err_b
        })
