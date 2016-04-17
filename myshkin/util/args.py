from docopt import docopt
from attrdict import AttrDict

def get_args(docstring):
    return AttrDict({k.replace("--", "").\
                       replace("<", "").\
                       replace(">", ""): v for k, v in docopt(docstring).iteritems()})
