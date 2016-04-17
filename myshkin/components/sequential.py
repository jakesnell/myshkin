class Sequential(object):
    def __init__(self, components):
        self.components = components

    def apply(self, x, train):
        out = x
        for component in self.components:
            out = component.apply(out, train)
        return out
