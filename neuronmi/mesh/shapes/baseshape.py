import numpy as np


class BaseShape(object):
    '''Common API of Neuron, Box, Shape'''
    _defaults = {}

    def __init__(self, params=None):
        if params is None:
            params = self.default_params

        self.conversion_factor = 1e4
            
        try:
            # Let's add the missing keys
            missing = set(self.default_params.keys()) - set(params.keys())
            # Extended
            params = params.copy()
            params.update(dict((key, self.default_params[key]) for key in missing))

            self.check_geometry_parameters(params)
        except NotImplementedError:
            print('No sanity checks ran on geometry inputs')

        self._params = params

    @property
    def default_params(self):
        return type(self)._defaults

    @property
    def params_cm(self):
        params = self._params
        for k in params.keys():
            if isinstance(params[k], (int, np.integer, float, np.float)):
                params[k] = params[k] * self.conversion_factor
        return params
        
    def contains(self, point, tol):
        '''Is point inside shape?'''
        return NotImplementedError

    def as_gmsh(self, model, tag=-1):
        '''Add shape to model (in terms of model.factory primitives)'''
        return NotImplementedError

    def link_surfaces(self, model, tags, links, box=None, tol=1E-10):
        '''
        Let tags be surfaces of the model. For every surface of the shape 
        we try to pair it with one of the tagged surfaces. 

        Return a map named_surface of shape -> tag. (updated links)
        
        NOTE: tags is altered in the process. 
        '''
        # Typically the linking depends just on the shape itself but
        # due to intersection in might be altered by the box.
        raise NotImplementedError

    # Generics ---------------------------------------------------------

    def bbox_contains(self, point, tol):
        '''Is point inside bounding box?'''
        return self._bbox.contains(point, tol)

    @property
    def bbox(self):
        '''Bounding box'''
        return self._bbox

    @property
    def control_points(self):
        '''Points characterizing the shape'''
        return self._control_points

    @property
    def center_of_mass(self):
        '''Center of mass is used to find the entity/its surfaces in the model'''
        return self._center_of_mass

    @property
    def surfaces(self):
        '''{str(name of the surface) -> R^3 that is its center of mass}'''
        return self._surfaces

