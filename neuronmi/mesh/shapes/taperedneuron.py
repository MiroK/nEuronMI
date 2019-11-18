from .utils import as_namedtuple, has_positive_values
from .gmsh_primitives import Sphere, Cylinder, Cone, Box
from .baseneuron import Neuron
from collections import OrderedDict
from math import sqrt
import numpy as np


class TaperedNeuron(Neuron):
    '''
    Neuron used e.g. in 

    Mainen, Zachary F., et al. 
    "A model of spike initiation in neocortical pyramidal neurons." 
    Neuron 15.6 (1995): 1427-1439.

    '''
    
    _defaults = {
        'soma_rad': 10,
        'soma_x': 0,
        'soma_y': 0,
        'soma_z': 0,  # Center
        'dendh_rad': 4,
        'dendh_len': 20,
        'dend_rad': 3,
        'dend_len': 50,
        'axonh_rad': 2,
        'axonh_len': 20,
        'axon_rad': 1,
        'axon_len': 50
        }
    
    def __init__(self, params=None):
        Neuron.__init__(self, params)

        # Define as Cylinder-Cone-Sphere-Cone-Cylinder
        # axon-axon hill-some-dend hill-dendrite
        params = as_namedtuple(self.params_cm)
        C = np.array([params.soma_x, params.soma_y, params.soma_z])
        # Move up
        shift = sqrt(params.soma_rad**2 - params.dendh_rad**2)
        
        D0 = C + np.array([0, 0, shift])
        D1 = D0 + np.array([0, 0, params.dendh_len])
        D2 = D1 + np.array([0, 0, params.dend_len])

        # Move down
        shift = sqrt(params.soma_rad**2 - params.axonh_rad**2)
        
        A0 = C - np.array([0, 0, shift])
        A1 = A0 - np.array([0, 0, params.axonh_len])
        A2 = A1 - np.array([0, 0, params.axon_len])

        self.pieces = OrderedDict(axon=Cylinder(A1, A2, params.axon_rad),
                                  axonh=Cone(A0, A1, params.axonh_rad, params.axon_rad),
                                  soma=Sphere(C, params.soma_rad),
                                  dendh=Cone(D0, D1, params.dendh_rad, params.dend_rad),
                                  dend=Cylinder(D1, D2, params.dend_rad))

        # Now draw circle around them
        self._control_points = np.row_stack([p.control_points for p in self.pieces.values()])
                               
        # Setup bounding box
        min_ = np.min(self._control_points, axis=0)
        max_ = np.max(self._control_points, axis=0)
        self._bbox = Box(min_, max_ - min_)

        # NOTE: missing center of gravity
        # The flat ones by center of mass
        self._surfaces = {k: self.pieces[k].center_of_mass for k in ('axon', 'soma', 'dend')}
        # Cone is special
        self._surfaces['axonh'] = self.pieces['axonh'].wall_center_of_mass
        self._surfaces['dendh'] = self.pieces['dendh'].wall_center_of_mass
        # Still missing end tips
        self._surfaces['axon_base'] = A2
        self._surfaces['dend_base'] = D2

    def check_geometry_parameters(self, params):
        assert set(params.keys()) == set(TaperedNeuron._defaults.keys()), \
            (set(TaperedNeuron._defaults.keys())-set(params.keys()), set(params.keys())-set(TaperedNeuron._defaults.keys()))
        # Ignore center
        assert has_positive_values(params,
                                   set(params.keys())-set(('soma_x', 'soma_y', 'soma_z')))

        assert params['soma_rad'] > params['dendh_rad'] > params['dend_rad']
        assert params['soma_rad'] > params['axonh_rad'] > params['axon_rad']
        
    def contains(self, point, tol):
        '''Is point inside shape?'''
        return any(piece.contains(point, tol) for piece in self.pieces.values())

    def as_gmsh(self, model, tag=-1):
        '''Add shape to model in terms of factory(gmsh) primitives'''
        soma = self.pieces['soma'].as_gmsh(model)
        axonh = self.pieces['axonh'].as_gmsh(model)
        axon = self.pieces['axon'].as_gmsh(model)
        dendh = self.pieces['dendh'].as_gmsh(model)
        dend = self.pieces['dend'].as_gmsh(model)
        
        neuron_tags, _ = model.occ.fuse([(3, soma)], [(3, axon), (3, axonh), (3, dend), (3, dendh)])

        model.occ.synchronize()

        surfs = model.getBoundary(neuron_tags)

        # Volume tag, surfaces tag
        return neuron_tags[0][1], [s[1] for s in surfs]
        
# --------------------------------------------------------------------

if __name__ == '__main__':

    neuron = TaperedNeuron({'dend_rad': 0.2, 'axon_rad': 0.3})

    import gmsh
    import sys

    model = gmsh.model
    factory = model.occ

    gmsh.initialize(sys.argv)
    gmsh.fltk.initialize()
    gmsh.option.setNumber("General.Terminal", 1)


    cone = neuron.pieces['dendh']
    v = cone.as_gmsh(model)
    
    # neuron.as_gmsh(model)
    factory.synchronize();

    model.mesh.generate(3)
    #model.mesh.refine()
    #model.mesh.setOrder(2)
    #model.mesh.partition(4)
    
    gmsh.write("neuron.msh")
    gmsh.fltk.run()
    gmsh.finalize()
