class Model(object):
    def __repr__(self):
        return "{:s}({:s})".format(self.__class__.__name__, self.opts)
