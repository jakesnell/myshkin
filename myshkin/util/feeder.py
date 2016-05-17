from collections import OrderedDict
from multiprocess.pool import ApplyResult
from pathos.pools import ProcessPool
from itertools import izip

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

class FeedIndexedArray(object):
    def __init__(self, index_arr, arr):
        assert index_arr.ndim == 1
        assert np.max(index_arr) < arr.shape[0]
        self.index_arr = index_arr
        self.arr = arr

    def get_num_examples(self):
        return self.index_arr.shape[0]

    def get_examples(self, indices):
        return self.arr[self.index_arr[indices]]

class BatchFetcher(object):
    def __init__(self, bindings, batch_inds, n_workers=0):
        self.bindings = bindings
        self.batch_inds = batch_inds
        self.n_workers = n_workers
        self.batches = {}

        if n_workers > 0:
            self.p = ProcessPool(self.n_workers)

    def fetch_async(self, ib):
        if ib not in self.batches:
            if self.n_workers > 0:
                self.batches[ib] = {k: p.apipe(v.get_examples, self.batch_inds[ib]) \
                                        if not isinstance(v, FeedRandomStream) \
                                        else v.get_examples(self.batch_inds[ib]) \
                                    for (k, v) in self.bindings.iteritems()}
            else:
                self.batches[ib] = {k: v.get_examples(self.batch_inds[ib]) \
                                    for (k, v) in self.bindings.iteritems()}

    def retrieve_batch(self, ib):
        self.fetch_async(ib)
        assert ib in self.batches

        return {k: v.get() if isinstance(v, ApplyResult) else v \
                for (k, v) in self.batches[ib].iteritems()}

    def del_batch(self, ib):
        del self.batches[ib]

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

    def feeds(self, shuffle=True, n_workers=0, look_ahead=5):
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

        batch_inds = [x for x in batch_ind_iter(n_examples)]
        batch_fetcher = BatchFetcher(self.bindings, batch_inds, n_workers=n_workers)

        feed_dicts = {}
        for ib in xrange(len(batch_inds)):
            for j in xrange(0, look_ahead+1):
                batch_fetcher.fetch_async(ib)

            feed_dict = batch_fetcher.retrieve_batch(ib)

            yield feed_dict

            batch_fetcher.del_batch(ib)

    def truncate(self, num_examples):
        assert self.get_num_examples() is None or num_examples <= self.get_num_examples()
        return Feeder(self.bindings,
                      batch_size=self.batch_size,
                      num_examples=num_examples)

    def bindings_keys(self):
        return self.bindings.keys()

class ZippedFeeder(object):
    def __init__(self, feeders):
        self.feeders = feeders

    def feeds(self, shuffle=True, **kwargs):
        if not isinstance(shuffle, list):
            shuffle = [shuffle] * len(self.feeders)

        assert len(shuffle) == len(self.feeders)

        for feed_t in izip(*[feeder.feeds(shuffle=sval, **kwargs)
                             for (feeder, sval) in zip(self.feeders, shuffle)]):

            feed_dict = {}
            for d in feed_t:
                feed_dict.update(d)

            yield feed_dict

    def truncate(self, num_examples):
        return ZippedFeeder([feeder.truncate(num_examples) for feeder in self.feeders])

    def bindings_keys(self):
        return list(set(sum([feeder.bindings_keys() for feeder in self.feeders], [])))

    def get_num_examples(self):
        finite_nexamples = filter(lambda x: x is not None,
                                  [feeder.get_num_examples()
                                   for feeder in self.feeders])
        assert len(finite_nexamples) > 0, "ZippedFeeder will never terminate"
        return np.min(finite_nexamples)

class RepeatedFeeder(object):
    def __init__(self, feeder, batch_size, num_examples=None):
        self.feeder = feeder
        self.batch_size = batch_size
        self.num_examples = num_examples

    def get_num_examples(self):
        return self.num_examples

    def feeds(self, **kwargs):
        def merge_feed(feed_dict, feed):
            if len(feed_dict.keys()) == 0:
                return feed
            else:
                assert set(feed_dict.keys()) == set(feed.keys())
                return {k: np.concatenate([v, feed[k]], axis=0)
                        for (k, v) in feed_dict.iteritems()}

        n_remaining = self.num_examples

        feed_dict = {}
        while True:
            for feed in self.feeder.feeds(**kwargs):
                # process feed
                feed_dict = merge_feed(feed_dict, feed)

                # yield as much as we can
                n_target = min(self.batch_size, n_remaining) if n_remaining is not None else self.batch_size
                while feed_dict.values()[0].shape[0] >= n_target and (n_remaining is None or n_remaining > 0):
                    rval = {k: v[:n_target] for (k, v) in feed_dict.iteritems()}
                    yield rval
                    feed_dict = {k: v[n_target:] for (k, v) in feed_dict.iteritems()}
                    if n_remaining is not None:
                        n_remaining -= n_target
                    n_target = min(self.batch_size, n_remaining) if n_remaining is not None else self.batch_size

                assert n_remaining is None or n_remaining >= 0
                if n_remaining == 0:
                    break

    def truncate(self, num_examples):
        assert self.num_examples is None or num_examples <= self.num_examples
        return RepeatedFeeder(self.feeder, self.batch_size, num_examples=num_examples)

    def bindings_keys(self):
        return self.feeder.bindings_keys()

def reduce_batches(sess, bindings, feeder, updates=[], shuffle=True, verbose=False):
    # bindings: dict of label => tensor
    # feeder.bindings: dict of tensor => Feed
    compute_labels = [k for (k, v) in bindings.iteritems() if v not in feeder.bindings_keys()]
    feed_labels = [k for (k, v) in bindings.iteritems() if v in feeder.bindings_keys()]

    rval = OrderedDict({})

    for ib, feed in enumerate(feeder.feeds(shuffle=shuffle)):
        if len(compute_labels) >= 1:
            batch_result = sess.run([bindings[label] for label in compute_labels] + updates, feed)
        else:
            batch_result = []

        for (label, val) in zip(compute_labels, batch_result) + [(feed_label, feed[bindings[feed_label]]) for feed_label in feed_labels]:
            if val.shape == ():
                raise ValueError("cannot reduce scalars")
            else:
                rval[label] = np.concatenate([rval.get(label, np.empty((0,) + val.shape[1:])), val], axis=0)

        if verbose:
            print "[Batch {:d}]".format(ib)

    return rval
