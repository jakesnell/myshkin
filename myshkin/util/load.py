from attrdict import AttrDict
import importlib
import yaml

def set_nested_key(d, keys, v):
    if len(keys) == 1:
        d[keys[0]] = v
    else:
        set_nested_key(d[keys[0]], keys[1:], v)

    return None

def load_model(opts_file, model_module='myshkin.models', opts={}, **kwargs):
    with open(opts_file, 'r') as f:
        opts_dict = AttrDict(yaml.load(f))

    for (k, v) in opts.iteritems():
        set_nested_key(opts_dict['opts'], k.split('.'), v)

    return load_model_from_dict(opts_dict, model_module=model_module, **kwargs)

def load_model_from_dict(opts_dict, model_module='myshkin.models', **kwargs):
    mod = importlib.import_module(model_module)

    model_type = getattr(mod, opts_dict.model)
    opts_type = getattr(mod, opts_dict.model + 'Opts')

    return model_type(opts_type(**opts_dict.opts), **kwargs)

def load_learn_opts(opts_file):
    with open(opts_file, 'r') as f:
        return AttrDict(yaml.load(f))
