import tensorflow as tf

class Identity(object):
    def __init__(self):
        pass

    def apply(self, x):
        return x

    def __repr__(self):
        return "{:s}()".format(self.__class__.__name__)

class Relu(object):
    def __init__(self):
        pass

    def apply(self, x):
        return tf.nn.relu(x)

    def __repr__(self):
        return "{:s}()".format(self.__class__.__name__)

class Sigmoid(object):
    def __init__(self):
        pass

    def apply(self, x):
        return tf.nn.sigmoid(x)

    def __repr__(self):
        return "{:s}()".format(self.__class__.__name__)
