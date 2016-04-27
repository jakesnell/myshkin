"""train_mnist.py

Usage:
    train_mnist.py <modelconf> <learnconf> [--nodash]
"""
import os
from subprocess import call

import numpy as np
import tensorflow as tf

from keras import backend as K

from myshkin.callbacks import *
from myshkin.data.mnist import load_mnist
from myshkin.fit import fit
from myshkin.util.args import get_args
from myshkin.util.feeder import Feeder, FeedArray, FeedRandomStream
from myshkin.util.load import load_model, load_learn_opts
from myshkin.util.session import get_device, get_session

def main():
    args = get_args(__doc__)

    with get_session() as sess, get_device():
        tf.set_random_seed(1234)
        np.random.seed(1234)

        model = load_model(args.modelconf)
        learn_opts = load_learn_opts(args.learnconf)

        mnist_data = load_mnist()

        train_feeder = Feeder({
                model.view.x_bk: FeedArray(mnist_data.x_train),
                model.view.y_b: FeedArray(mnist_data.y_train),
                K.learning_phase(): FeedRandomStream(lambda b: 1)
            },
            batch_size=learn_opts.batch_size
        )

        valid_feeder = Feeder({
                model.view.x_bk: FeedArray(mnist_data.x_valid),
                model.view.y_b: FeedArray(mnist_data.y_valid),
                K.learning_phase(): FeedRandomStream(lambda b: 0)
            },
            batch_size=learn_opts.batch_size
        )

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

        callbacks = [DefaultLogger(['loss', 'err']),
                     DeepDashboardLogger(
                         learn_opts.exp_id,
                         [RawLogger('out.log', 'Log', ['loss', 'err']),
              CSVLogger('loss.csv', 'Loss', ['loss']),
                          CSVLogger('err.csv', 'Classification Error', ['err'])]
                     ),
                     EarlyStopping('loss', learn_opts.patience),
                     ModelCheckpoint(os.path.join(os.getenv('MYSHKIN_CHECKPOINTDIR', '.'),
                                                  learn_opts.exp_id),
                                     'loss',
                                     verbose=True)]

        if not args.nodash:
            call(["open", "http://localhost/deep-dashboard/?id={:s}".format(learn_opts.exp_id)])
        else:
            print "experiment id: {:s}".format(learn_opts.exp_id)

        fit(model,
            optimizer,
            train_feeder,
            valid_feeder,
            sess,
            callbacks=callbacks,
            n_epochs=learn_opts.n_epochs)

if __name__ == '__main__':
    main()
