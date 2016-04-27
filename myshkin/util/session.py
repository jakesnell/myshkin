import os

import tensorflow as tf
from keras import backend as K

def get_device():
    tf_device = os.getenv('TENSORFLOW_DEVICE', '/cpu:0')
    print "using device {:s}".format(tf_device)

    return tf.device(tf_device)

def get_session():
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    K.set_session(sess)

    return sess
