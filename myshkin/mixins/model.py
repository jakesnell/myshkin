import yaml

class Model(object):
    def save_conf(self, out_file):
        with open(out_file, 'w') as f:
            conf_dict = {'model': self.__class__.__name__,
                         'opts': dict(self.opts._asdict())}
            f.write(yaml.dump(conf_dict, default_flow_style=False))

    def __repr__(self):
        return "{:s}({:s})".format(self.__class__.__name__, self.opts)

