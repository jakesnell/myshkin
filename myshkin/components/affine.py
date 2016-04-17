import numpy as np
import tensorflow as tf

class Affine(object):
    def __init__(self, input_size, output_size, name=None):
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.name = name

        with tf.name_scope(name):
            self.weight = tf.Variable(
                              tf.truncated_normal([
                                  self.input_size,
                                  self.output_size
                              ], stddev=1.0/np.sqrt(float(self.input_size))),
                              name='weight'
                          )
            self.bias = tf.Variable(
                            tf.zeros([self.output_size]),
                            name='bias'
                        )

    def apply(self, x, train):
        return tf.matmul(x, self.weight) + self.bias

    def __repr__(self):
        return "{:s}({:d}, {:d}, name='{:s}')".format(self.__class__.__name__,
                                                      self.input_size,
                                                      self.output_size,
                                                      self.name)
