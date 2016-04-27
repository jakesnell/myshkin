import numpy as np
import tensorflow as tf

#class Convolution2D(object):
#    def __init__(self, input_depth, n_filters, kernel_shape, subsample_shape,
#                       border_mode='SAME', name=None):
#        assert len(kernel_shape) == 2
#        assert len(subsample_shape) == 2
#
#        self.input_depth = input_depth
#        self.n_filters = n_filters
#        self.kernel_shape = kernel_shape
#        self.subsample_shape = subsample_shape 
#        self.border_mode = border_mode
#        self.name = name
#
#        with tf.name_scope(name):
#            self.kernel = tf.Variable(
#                              tf.truncated_normal([
#                                  self.kernel_shape[0],
#                                  self.kernel_shape[1],
#                                  self.input_depth,
#                                  self.n_filters
#                              ], stddev=1.0/np.sqrt(float(self.input_depth * self.kernel_shape[0] * self.kernel_shape[1]))),
#                              name='kernel'
#                          )
#            self.bias = tf.Variable(
#                            tf.zeros([self.n_filters]),
#                            name='bias'
#                        )
#
#    def apply(self, x):
#        return tf.nn.conv2d(x, self.kernel, [1] + list(self.subsample_shape) + [1], padding=self.border_mode,)
#
#    def __repr__(self):
#        return "{:s}({:d}, {:d}, {:s}, {:s}, border_mode='{:s}', name='{:s}')".format(
#                self.__class__.__name__,
#                self.input_depth,
#                self.n_filters,
#                self.kernel_shape,
#                self.subsample_shape,
#                self.border_mode,
#                self.name)

class MaxPool2D(object):
    def __init__(self):
        pass

    def apply(self, x):
        return tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    def __repr__(self):
        return "{:s}()".format(self.__class__.__name__)
