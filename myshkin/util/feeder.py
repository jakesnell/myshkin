from collections import OrderedDict
from multiprocess.pool import ApplyResult
from pathos.pools import ProcessPool

import numpy as np

class FeedArray(object):
    def __init__(self, arr):
        self.arr = arr

    def get_num_examples(self):
        return self.arr.shape[0]

    def get_examples(self, indices):
        return self.arr[indices]

class FeedRandomStream(object):
    def __init__(self, fun, n_examples=None):
        # fun: function from batch_size => random array
        self.fun = fun
        self.n_examples = n_examples

    def get_num_examples(self):
        return self.n_examples

    def get_examples(self, indices):
        return self.fun(len(indices))

class Feeder(object):
    def __init__(self, bindings, batch_size=128, num_examples=None):
        self.bindings = bindings
        self.batch_size = batch_size
        self.num_examples = num_examples

    def get_num_examples(self):
        if self.num_examples is None:
            return self.bindings.values()[0].get_num_examples()
        else:
            return self.num_examples

    def feeds(self, shuffle=True, include_size=False, n_workers=4, look_ahead=5):
        p = ProcessPool(n_workers)

        n_examples = self.get_num_examples()
        indices = np.arange(n_examples)

        if shuffle:
            np.random.shuffle(indices)

        def batch_ind_iter(n_examples):
            ind = 0
            while ind < n_examples:
                batch_end = min(ind + self.batch_size, n_examples)
                batch_inds = indices[ind:batch_end]
                yield batch_inds
                ind = batch_end

        batches = [x for x in batch_ind_iter(n_examples)]
        feed_dicts = {}
        for ib in xrange(len(batches)):
            if ib not in feed_dicts:
                feed_dicts[ib] = {k: p.apipe(v.get_examples, batches[ib]) \
                                    if not isinstance(v, FeedRandomStream) \
                                    else v.get_examples(batches[ib]) \
                                  for (k, v) in self.bindings.iteritems()}

            for j in xrange(1, look_ahead+1):
                if ib + j < len(batches) and (ib + j) not in feed_dicts:
                    feed_dicts[ib+j] = {k: p.apipe(v.get_examples, batches[ib+j]) \
                                           if not isinstance(v, FeedRandomStream) \
                                           else v.get_examples(batches[ib+j]) \
                                        for (k, v) in self.bindings.iteritems()}

            feed_dict = {k: v.get() if isinstance(v, ApplyResult) else v \
                         for (k, v) in feed_dicts[ib].iteritems()}

            if include_size:
                yield len(batches[ib]), feed_dict
            else:
                yield feed_dict

            del feed_dicts[ib]

    def truncate(self, num_examples):
        assert self.get_num_examples() is None or num_examples <= self.get_num_examples()
        return Feeder(self.bindings,
                      batch_size=self.batch_size,
                      num_examples=num_examples)

def reduce_batches(sess, bindings, feeder, updates=[], shuffle=True, verbose=False):
    # bindings: dict of label => tensor
    # feeder.bindings: dict of tensor => Feed
    compute_labels = [k for (k, v) in bindings.iteritems() if v not in feeder.bindings.keys()]
    feed_labels = [k for (k, v) in bindings.iteritems() if v in feeder.bindings.keys()]

    rval = OrderedDict({})

    n_examples = 0
    for bs, feed in feeder.feeds(include_size=True, shuffle=shuffle):
        if len(compute_labels) >= 1:
            batch_result = sess.run([bindings[label] for label in compute_labels] + updates, feed)
        else:
            batch_result = []

        for (label, val) in zip(compute_labels, batch_result) + [(feed_label, feed[bindings[feed_label]]) for feed_label in feed_labels]:
            if val.shape == ():
                rval[label] = rval.get(label, 0.0) + bs * val
            else:
                rval[label] = np.concatenate([rval.get(label, np.empty((0,) + val.shape[1:])), val], axis=0)

        n_examples += bs

        if verbose:
            disp_strs = []
            for label in rval.keys():
                if rval[label].shape == ():
                    disp_strs.append("{:s} = {:0.6f}".format(label, 1.0 * rval[label] / n_examples))

            print "[{:d}/{:d}] {:s}".format(n_examples, feeder.get_num_examples(), ", ".join(disp_strs))

    for label in compute_labels + feed_labels:
        if rval[label].shape == ():
            rval[label] /= n_examples

    return rval
