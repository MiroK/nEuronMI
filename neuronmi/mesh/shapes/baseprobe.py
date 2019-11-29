from neuronmi.mesh.shapes.baseshape import BaseShape


class Probe(BaseShape):
    '''Every probe'''
    def __init__(self, params=None):
        BaseShape.__init__(self, params)

    def check_geometry_parameters(self, params):
        '''
        Params contains all params as given by defaults and their values
        are sane.
        '''
        raise NotImplementedError
