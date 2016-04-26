import tensorflow as tf

class Reshape(object):
    def __init__(self, dest_shape):
        self.dest_shape = dest_shape

    def apply(self, x, train):
        return tf.reshape(x, self.dest_shape)

    def __repr__(self):
        return "{:s}({:s})".format(self.__class__.__name__, self.dest_shape)
