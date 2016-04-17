from operator import add

import tensorflow as tf

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

def fit(model, optimizer, train_feeder, valid_feeder, n_epochs=100, callbacks=[]):
    monitor_fields = list(set(reduce(add, [callback.get_monitor_fields() for callback in callbacks])))

    train_step = optimizer.minimize(model.train_view.loss)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        train_fields = {field: model.train_view[field] for field in monitor_fields}
        valid_fields = {field: model.test_view[field] for field in monitor_fields}

        for ie in xrange(n_epochs):
            train_stats = reduce_batches(sess, train_fields, train_feeder, updates=[train_step])
            valid_stats = reduce_batches(sess, valid_fields, valid_feeder)

            for callback in callbacks:
                callback.epoch_end(ie, model, train_stats, valid_stats)

            if any([callback.stop_training() for callback in callbacks]):
                break

        for callback in callbacks:
            if callback.stop_training():
                print callback.stop_training_message()
