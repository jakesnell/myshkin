from attrdict import AttrDict
import importlib
import yaml

def load_model(opts_file, model_module='myshkin.models', **kwargs):
    mod = importlib.import_module(model_module)

    with open(opts_file, 'r') as f:
        opts_dict = AttrDict(yaml.load(f))

    model_type = getattr(mod, opts_dict.model)
    opts_type = getattr(mod, opts_dict.model + 'Opts')

    return model_type(opts_type(**opts_dict.opts), **kwargs)

def load_learn_opts(opts_file):
    with open(opts_file, 'r') as f:
        return AttrDict(yaml.load(f))
