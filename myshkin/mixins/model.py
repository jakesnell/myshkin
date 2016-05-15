import os
import glob
import yaml
import numpy as np

import keras

class Model(object):
    def save_conf(self, out_file):
        with open(out_file, 'w') as f:
            conf_dict = {'model': self.__class__.__name__,
                         'opts': dict(self.opts._asdict())}
            f.write(yaml.dump(conf_dict, default_flow_style=False))

    def save_weights(self, out_dir, verbose=False):
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        def _save_weights(out_dir, name, component):
            if isinstance(component, keras.models.Model):
                if verbose:
                    print "saving {:s}...".format(name)
                component.save_weights(os.path.join(out_dir, name + ".h5"), overwrite=True)
            else:
                for k, subcomponent in component.components.iteritems():
                    _save_weights(out_dir, name + "." + k, subcomponent)

        for k, component in self.components.iteritems():
            _save_weights(out_dir, k, component)

    def get_component(self, specs):
        cur = self
        for spec in specs:
            cur = cur.components[spec]
        return cur

    def load_weights(self, weights_dir, verbose=False):
        weight_files = glob.glob(os.path.join(weights_dir, '*.h5'))
        for weight_file in weight_files:
            component = self.get_component(os.path.basename(weight_file).split(".")[:-1])
            if verbose:
                print "loading from {:s}...".format(os.path.basename(weight_file))
            component.load_weights(weight_file)

    def __repr__(self):
        return "{:s}({:s})".format(self.__class__.__name__, self.opts)
