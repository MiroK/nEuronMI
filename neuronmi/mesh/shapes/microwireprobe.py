from neuronmi.mesh.shapes.utils import as_namedtuple, has_positive_values, link_surfaces
from neuronmi.mesh.shapes.gmsh_primitives import Cylinder
from neuronmi.mesh.shapes.baseprobe import Probe
import numpy as np


class MicrowireProbe(Probe):
    '''
    Z-axis aligned cylinder specified by (x, y, z) of the endpoint, radius
    and length
    '''

    _defaults = {
        'tip_x': 50,
        'tip_y': 0,
        'tip_z': 0,
        'radius': 10,
        'length': 1000
        }
    
    def __init__(self, params=None):
        Probe.__init__(self, params)
        
        params = as_namedtuple(self.params)
        A = np.array([params.tip_x, params.tip_y, params.tip_z])
        B = A + np.array([0, 0, params.length])
        self.cylinder = Cylinder(A, B, params.radius)
        
        # Now draw circle around them
        self._control_points = self.cylinder.control_points
                               
        # Setup bounding box
        self._bbox = self.cylinder.bbox

        # NOTE: missing center of gravity
        self._surfaces = {'wall': self.cylinder.center_of_mass,
                          'contact_0': A}

    def check_geometry_parameters(self, params):
        assert set(params.keys()) == set(MicrowireProbe._defaults.keys()) 
        # Ignore center
        assert has_positive_values(params,
                                   set(params.keys()) - set(('tip_x', 'tip_y', 'tip_z')))
        
    def contains(self, point, tol):
        '''Is point inside shape?'''
        return self.cylinder.contains(point, tol)

    def as_gmsh(self, model, tag=-1):
        '''Add shape to model in terms of factory(gmsh) primitives'''
        probe = self.cylinder.as_gmsh(model, tag)
        
        return [probe]

    def link_surfaces(self, model, tags, links, box, tol=1E-10):
        '''Account for possible cut and shift of center of mass of face'''
        # Should be fine for tip
        links = link_surfaces(model, tags, self, links=links, tol=tol)
        # NOTE: as we chop the by box, the wall won't be found with the
        # above metric; But we should match x, y and z should account for chop
        Z0 = 0.5*(box.max_[2] + self.params['tip_z'])
        metric = lambda x, y: np.sqrt(np.abs((y - x)[:, 0])**2 + np.abs((y - x)[:, 1])**2 + np.abs((x[:, 2]-Z0)**2))
        return link_surfaces(model, tags, self, links=links, metric=metric, tol=tol)

# --------------------------------------------------------------------

if __name__ == '__main__':

    neuron = MicrowireProbe()

    import gmsh
    import sys

    model = gmsh.model
    factory = model.occ

    gmsh.initialize(sys.argv)

    gmsh.option.setNumber("General.Terminal", 1)

    neuron.as_gmsh(model)
    factory.synchronize();

    model.mesh.generate(3)
    #model.mesh.refine()
    #model.mesh.setOrder(2)
    #model.mesh.partition(4)
    
    gmsh.write("probe.msh")
    
    gmsh.finalize()
