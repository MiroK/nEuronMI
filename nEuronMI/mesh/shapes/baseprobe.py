from baseshape import BaseShape


class Probe(object):
    '''Every probe'''
    def __init__(self, params=None):
        if params is None:
            params = self.default_params
            
        try:
            self.check_geometry_parameters(params)
        except NotImplementedError:
            print('No sanity checks ran on geometry inputs')

        self._params = params
        
    @property
    def default_params(self):
        return type(self)._defaults

    def check_geometry_parameters(self, params):
        '''
        Params contains all params as given by defaults and their values
        are sane.
        '''
        raise NotImplementedError
