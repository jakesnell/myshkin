import numpy as np

class Feeder(object):
    def __init__(self, bindings, batch_size=128):
        self.bindings = bindings
        self.batch_size = batch_size

    def feeds(self, shuffle=True, include_size=False):
        n_examples = self.bindings.values()[0].shape[0]
        indices = np.arange(n_examples)

        if shuffle:
            np.random.shuffle(indices)

        ind = 0
        while ind < n_examples:
            batch_end = min(ind + self.batch_size, n_examples)
            batch_inds = indices[ind:batch_end]
            feed_dict = {k: v[batch_inds] for (k, v) in self.bindings.iteritems()}
            if include_size:
                yield batch_end - ind, feed_dict
            else:
                yield feed_dict
            ind = batch_end

def reduce_batches(sess, bindings, feeder, updates=[]):
    labels = bindings.keys()
    rval = {k: 0.0 for k in labels}

    n_examples = 0
    for bs, feed in feeder.feeds(include_size=True):
        batch_result = sess.run([bindings[label] for label in labels] + updates, feed)
        for (label, val) in zip(labels, batch_result):
            rval[label] += bs * val
        n_examples += bs

    for label in labels:
        rval[label] /= n_examples

    return rval
