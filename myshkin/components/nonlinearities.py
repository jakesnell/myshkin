import tensorflow as tf

class Relu(object):
    def __init__(self):
        pass

    def apply(self, x, train):
        return tf.nn.relu(x)

    def __repr__(self):
        return "{:s}()".format(self.__class__.__name__)
