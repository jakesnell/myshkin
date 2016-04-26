from attrdict import AttrDict
from collections import namedtuple

import tensorflow as tf

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
    def __init__(self, opts, name=None):
        if name is None:
            name = self.__class__.__name__

        with tf.name_scope(name):
            self.opts = opts

            self.x_bhh = tf.placeholder(tf.float32,
                                        shape=[None, self.opts.h, self.opts.h],
                                        name='x_bhh')
            self.y_b = tf.placeholder(tf.int32,
                                      shape=[None],
                                      name='y_b')

            layers = []
            cur_dim = 1
            layer_ind = 0
            for next_dim in opts.conv_dims:
                layers.append(Convolution2D(cur_dim, next_dim, (5, 5), (1, 1),
                                                name="conv_layer{:d}".format(layer_ind),
                                                border_mode="SAME"))
                layers.append(Relu())
                layers.append(MaxPool2D())
                cur_dim = next_dim
                layer_ind += 1

            fc_dim = cur_dim * (self.opts.h ** 2) // (4 ** len(self.opts.conv_dims))
            layers.append(Reshape((-1, fc_dim)))

            layer_ind = 0
            cur_dim = fc_dim
            for next_dim in opts.hid_dims:
                layers.append(Affine(cur_dim, next_dim, name="fc_layer{:d}".format(layer_ind)))
                layers.append(Relu())
                layers.append(Dropout(self.opts.dropout))
                cur_dim = next_dim
                layer_ind += 1

            layers.append(Affine(cur_dim, self.opts.c, name="fc_layer{:d}".format(layer_ind)))

            self.log_classifier = Sequential(layers)

            self.train_view = self.build(self.x_bhh, True)
            self.test_view = self.build(self.x_bhh, False)

    def build(self, x_bhh, train):
        x_bhh1 = tf.reshape(x_bhh, [-1, self.opts.h, self.opts.h, 1])
        log_classifier_seq = self.log_classifier.apply_seq(x_bhh1, train)
        log_y_hat_bc = log_classifier_seq[-1]
        loss_b = tf.nn.sparse_softmax_cross_entropy_with_logits(log_y_hat_bc, self.y_b)
        loss = tf.reduce_mean(loss_b)

        y_hat_bc = tf.nn.softmax(log_y_hat_bc)

        acc_b = tf.equal(tf.cast(tf.argmax(y_hat_bc, 1), tf.int32), self.y_b)
        acc = tf.reduce_mean(tf.cast(acc_b, tf.float32))
        err = 1.0 - acc

        return AttrDict({
            'x_bhh': x_bhh,
            'y_b': self.y_b,
            'y_hat_bc': y_hat_bc,
            'z_bm': log_classifier_seq[-2],
            'loss': loss,
            'acc': acc,
            'err': err
        })

