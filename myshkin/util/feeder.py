import numpy as np

class FeedArray(object):
    def __init__(self, arr):
        self.arr = arr

    def get_num_examples(self):
        return self.arr.shape[0]

    def get_examples(self, indices):
        return self.arr[indices]

class FeedRandomStream(object):
    def __init__(self, fun):
        # fun: function from batch_size => random array
        self.fun = fun

    def get_num_examples(self):
        return None

    def get_examples(self, indices):
        return self.fun(len(indices))

class Feeder(object):
    def __init__(self, bindings, batch_size=128):
        self.bindings = bindings
        self.batch_size = batch_size

    def feeds(self, shuffle=True, include_size=False):
        n_examples = self.bindings.values()[0].get_num_examples()
        indices = np.arange(n_examples)

        if shuffle:
            np.random.shuffle(indices)

        ind = 0
        while ind < n_examples:
            batch_end = min(ind + self.batch_size, n_examples)
            batch_inds = indices[ind:batch_end]
            feed_dict = {k: v.get_examples(batch_inds) for (k, v) in self.bindings.iteritems()}
            if include_size:
                yield batch_end - ind, feed_dict
            else:
                yield feed_dict
            ind = batch_end

def reduce_batches(sess, bindings, feeder, updates=[], shuffle=True):
    # bindings: dict of label => tensor
    # feeder.bindings: dict of tensor => Feed
    compute_labels = [k for (k, v) in bindings.iteritems() if v not in feeder.bindings.keys()]
    feed_labels = [k for (k, v) in bindings.iteritems() if v in feeder.bindings.keys()]

    rval = {}

    n_examples = 0
    for bs, feed in feeder.feeds(include_size=True, shuffle=shuffle):
        batch_result = sess.run([bindings[label] for label in compute_labels] + updates, feed)
        for (label, val) in zip(compute_labels, batch_result) + [(feed_label, feed[bindings[feed_label]]) for feed_label in feed_labels]:
            if val.shape == ():
                rval[label] = rval.get(label, 0.0) + bs * val
            else:
                rval[label] = np.concatenate([rval.get(label, np.empty((0,) + val.shape[1:])), val], axis=0)

        n_examples += bs

    for label in compute_labels + feed_labels:
        if rval[label].shape == ():
            rval[label] /= n_examples

    return rval
