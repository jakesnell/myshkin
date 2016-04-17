"""test_mnist.py

Usage:
    test_mnist.py <checkpointdir>
"""
import os
import yaml

import numpy as np
import tensorflow as tf

from myshkin.data.mnist import load_mnist
from myshkin.evaluate import evaluate
from myshkin.util.args import get_args
from myshkin.util.feeder import Feeder
from myshkin.util.load import load_model, load_learn_opts

def main():
    args = get_args(__doc__)

    tf_device = os.getenv('TENSORFLOW_DEVICE', '/cpu:0')
    print "using device {:s}".format(tf_device)

    with tf.device(tf_device):
        tf.set_random_seed(1234)
        np.random.seed(1234)

        model = load_model(os.path.join(args.checkpointdir, 'model_conf.yaml'))

        mnist_data = load_mnist()

        test_feeder = Feeder({
                model.test_view.x_bk: mnist_data.x_test,
                model.test_view.y_b: mnist_data.y_test
            },
            batch_size=128
        )

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            # restore model parameters
            checkpoint_file = os.path.join(args.checkpointdir, 'checkpoint')
            with open(checkpoint_file, 'r') as f:
                param_file = os.path.join(args.checkpointdir,
                                          yaml.load(f)['model_checkpoint_path'])

            saver.restore(sess, param_file)

            test_fields = ['loss', 'err']
            test_stats = evaluate(sess, model, test_feeder, test_fields)
            print ", ".join(["test {:s} = {:0.8f}".format(field, test_stats[field])
                             for field in test_fields])

if __name__ == '__main__':
    main()
