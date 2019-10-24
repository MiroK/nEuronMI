from baseshape import BaseShape
import utils


class Neuron(BaseShape):
    '''Every neuron'''
    # NOTE: we assume that its axis is aligned with Z axis
    def __init__(self, params, tol):
        '''Tol is a tolerance for finite precision arithmetic of gmsh'''
        BaseShape.__init__(self, params)

    def link_surfaces(self, model, tags, links, box=None, tol=1E-10):
        '''Rely on correspondence of center of mass'''
        return utils.link_surfaces(model, tags, self, links, tol=tol)
        
    @property
    def default_params(self):
        return type(self)._defaults

    def check_geometry_parameters(self, params):
        '''
        Params contains all params as given by defaults and their values
        are sane.
        '''
        raise NotImplementedError
