from neuronmi.mesh.shapes.utils import as_namedtuple, has_positive_values
from neuronmi.mesh.shapes.gmsh_primitives import Sphere, Cylinder, Cone, Box
from neuronmi.mesh.shapes.baseneuron import Neuron
from collections import OrderedDict
from math import sqrt
import numpy as np


class BallStickNeuron(Neuron):
    '''Soma(sphere) with cylinders as axon/dendrite.'''
    _defaults = {
        'soma_rad': 10,
        'soma_x': 0,
        'soma_y': 0,
        'soma_z': 0,  # Center
        'dend_rad': 2.5,
        'dend_len': 200,
        'axon_rad': 2,
        'axon_len': 100
        }    
    
    def __init__(self, params=None):
        Neuron.__init__(self, params)

        # Define as Cylinder-Sphere-Cylinder
        # axon-axon hill-some-dend hill-dendrite
        params = as_namedtuple(self._params)
        C = np.array([params.soma_x, params.soma_y, params.soma_z])
        # Move up
        shift = sqrt(params.soma_rad**2 - params.dend_rad**2)

        D0 = C + np.array([0, 0, shift])
        D1 = D0 + np.array([0, 0, params.dend_len])

        # Move down
        shift = sqrt(params.soma_rad**2 - params.axon_rad**2)

        A0 = C - np.array([0, 0, shift])
        A1 = A0 - np.array([0, 0, params.axon_len])

        self.pieces = OrderedDict(axon=Cylinder(A0, A1, params.axon_rad),
                                  soma=Sphere(C, params.soma_rad),
                                  dend=Cylinder(D0, D1, params.dend_rad))

        # Now draw circle around them
        self._control_points = np.row_stack([p.control_points for p in self.pieces.values()])

        # Setup bounding box
        min_ = np.min(self._control_points, axis=0)
        max_ = np.max(self._control_points, axis=0)
        self._bbox = Box(min_, max_ - min_)

        # NOTE: missing center of gravity
        # Surfaces are associated with centers of the pieces; this way
        # we find walls
        self._surfaces = {k: self.pieces[k].center_of_mass for k in self.pieces}
        # Still missing end tips
        self._surfaces['axon_base'] = A1
        self._surfaces['dend_base'] = D1

    def check_geometry_parameters(self, params):
        assert set(params.keys()) == set(BallStickNeuron._defaults.keys())
        # Ignore center
        assert has_positive_values(params,
                                   set(params.keys())-set(('soma_x', 'soma_y', 'soma_z')))

        assert params['soma_rad'] > params['dend_rad']
        assert params['soma_rad'] > params['axon_rad']
        
    def contains(self, point, tol):
        '''Is point inside shape?'''
        return any(piece.contains(point, tol) for piece in self.pieces.values())

    def as_gmsh(self, model, tag=-1):
        '''Add shape to model in terms of factory(gmsh) primitives'''
        soma = self.pieces['soma'].as_gmsh(model)
        axon = self.pieces['axon'].as_gmsh(model)
        dend = self.pieces['dend'].as_gmsh(model)

        model.occ.synchronize()
                
        neuron_tags, _ = model.occ.fuse([(3, soma)], [(3, axon), (3, dend)])

        model.occ.synchronize()

        surfs = model.getBoundary(neuron_tags)
        
        # Volume tag, surfaces tag
        return neuron_tags[0][1], [s[1] for s in surfs]
