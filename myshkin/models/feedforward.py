from attrdict import AttrDict
from collections import namedtuple

import tensorflow as tf

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
            layers.append(Affine(cur_dim, next_dim, name="layer{:d}".format(layer_ind)))
            layers.append(Relu())
            layers.append(Dropout(self.opts.dropout))
            cur_dim = next_dim
            layer_ind += 1

        layers.append(Affine(cur_dim, self.opts.c, name="layer{:d}".format(layer_ind)))

        self.log_classifier = Sequential(layers)

        self.train_view = self.build(True)
        self.test_view = self.build(False)

    def build(self, train):
        log_y_hat_bc = self.log_classifier.apply(self.x_bk, train)
        loss_b = tf.nn.sparse_softmax_cross_entropy_with_logits(log_y_hat_bc, self.y_b)
        loss = tf.reduce_mean(loss_b)

        y_hat_bc = tf.nn.softmax(log_y_hat_bc)

        acc_b = tf.equal(tf.cast(tf.argmax(y_hat_bc, 1), tf.int32), self.y_b)
        acc = tf.reduce_mean(tf.cast(acc_b, tf.float32))
        err = 1.0 - acc

        return AttrDict({
            'x_bk': self.x_bk,
            'y_b': self.y_b,
            'y_hat_bc': y_hat_bc,
            'loss': loss,
            'acc': acc,
            'err': err
        })
