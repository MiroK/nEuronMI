from neuronmi.mesh.shapes.baseshape import BaseShape
from neuronmi.mesh.shapes.utils import link_surfaces


class Neuron(BaseShape):
    '''Every neuron'''
    # NOTE: we assume that its axis is aligned with Z axis
    def __init__(self, params):
        BaseShape.__init__(self, params)

    def link_surfaces(self, model, tags, links, box=None, tol=1E-10):
        '''Rely on correspondence of center of mass'''
        return link_surfaces(model, tags, self, links, tol=tol)

    def check_geometry_parameters(self, params):
        '''
        Params contains all params as given by defaults and their values
        are sane.
        '''
        raise NotImplementedError
