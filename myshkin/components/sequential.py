class Sequential(object):
    def __init__(self, components):
        self.components = components

    def apply(self, x, train):
        return self.apply_seq(x, train)[-1]

    def apply_seq(self, x, train):
        rval = [x]
        for component in self.components:
            rval.append(component.apply(rval[-1], train))

        return rval[1:]

    def __repr__(self):
        return "{:s}([{:s}])".format(self.__class__.__name__,
                                     ", ".join([repr(component) for component in self.components]))
