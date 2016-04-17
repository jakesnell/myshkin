import tensorflow as tf

class Dropout(object):
    def __init__(self, keep_prob):
        self.keep_prob = keep_prob

    def apply(self, x, train):
        if train:
            return tf.nn.dropout(x, self.keep_prob)
        else:
            return x

    def __repr__(self):
        return "{:s}({:f})".format(self.__class__.__name__, self.keep_prob)
