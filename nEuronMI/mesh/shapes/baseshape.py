class BaseShape(object):
    '''Common API of Neuron, Box, Shape'''
    def contains(self, point, tol):
        '''Is point inside shape?'''
        return NotImplementedError

    def as_gmsh(self, factory, tag=-1):
        '''Add shape to model in terms of factory(gmsh) primitives'''
        return NotImplementedError
    
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

    
