from operator import add

import tensorflow as tf

from myshkin.util.feeder import reduce_batches

def fit(model, optimizer, train_feeder, valid_feeder, sess, n_epochs=100, callbacks=[], train_vars=None, verbose=False):
    monitor_fields = list(set(reduce(add, [callback.get_monitor_fields() for callback in callbacks])))

    ext_vars = set(tf.all_variables())
    if train_vars is None:
        train_step = optimizer.minimize(model.view.loss)
    else:
        train_step = optimizer.minimize(model.view.loss, var_list=train_vars)
    sess.run(tf.initialize_variables(set(tf.all_variables()) - ext_vars))

    for callback in callbacks:
        callback.register_sess(sess)

    fields = {field: model.view[field] for field in monitor_fields}

    for ie in xrange(n_epochs):
        train_stats = reduce_batches(sess, fields, train_feeder, updates=[train_step], verbose=verbose)
        valid_stats = reduce_batches(sess, fields, valid_feeder, verbose=verbose)

        for callback in callbacks:
            callback.epoch_end(ie, sess, model, train_stats, valid_stats)

        if any([callback.stop_training() for callback in callbacks]):
            break

    for callback in callbacks:
        if callback.stop_training():
            print callback.stop_training_message()
        callback.training_end()
