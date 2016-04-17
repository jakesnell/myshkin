from myshkin.util.feeder import reduce_batches

def evaluate(sess, model, feeder, eval_fields):
    test_fields = {field: model.test_view[field] for field in eval_fields}

    return reduce_batches(sess, test_fields, feeder)
