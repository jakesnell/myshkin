import tensorflow as tf

class ResizeNearestNeighbor(object):
    def __init__(self, new_shape):
        self.new_shape = new_shape

    def apply(self, x, train):
        return tf.image.resize_nearest_neighbor(x, self.new_shape)

    def __repr__(self):
        return "{:s}({:s})".format(self.__class__.__name__, self.new_shape)
