class Sequential(object):
    def __init__(self, components):
        self.components = components

    def apply(self, x, train):
        return self.apply_seq(x, train)[-1]

    def apply_seq(self, x):
        rval = [x]
        for component in self.components:
            if hasattr(component, 'apply'):
                rval.append(component.apply(rval[-1]))
            else:
                # Keras component
                rval.append(component(rval[-1]))

        return rval[1:]

    def __repr__(self):
        return "{:s}([{:s}])".format(self.__class__.__name__,
                                     ", ".join([repr(component) for component in self.components]))
