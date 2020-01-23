from neuronmi.mesh.shapes.baseshape import BaseShape


class Probe(BaseShape):
    '''Every probe'''
    def __init__(self, params=None):
        BaseShape.__init__(self, params)

    @staticmethod
    def check_geometry_parameters(params):
        '''
        Params contains all params as given by defaults and their values
        are sane.
        '''
        raise NotImplementedError

    @staticmethod
    def get_probe_type():
        raise NotImplementedError

    def get_electrode_centers(self, unit='um'):
        '''
        Returns centers of electrodes

        Parameters
        ----------
        unit: str
            'um' or 'cm'

        Returns
        -------
        contacts: np.array
            Array with electrode centers
        '''
        if unit == 'um':
            return self._contacts
        elif unit == 'cm':
            return self._contacts * 1e-4
