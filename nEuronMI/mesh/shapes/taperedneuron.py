from utils import as_namedtuple, has_positive_values
from baseshape import BaseShape
from baseneuron import Neuron
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
        'soma_rad': 1,
        'soma_x': 0,
        'soma_y': 0,
        'soma_z': 0,  # Center
        'dendh_rad': 0.5,
        'dendh_len': 1,
        'dend_rad': 0.4,
        'dend_len': 1,
        'axonh_rad': 0.5,
        'axonh_len': 1,
        'axon_rad': 0.4,
        'axon_len': 1
        }
    
    def __init__(self, params=None):
        Neuron.__init__(self, params)

        # TODO: Setup bounding box
        # TODO: Setup control points

    def check_geometry_parameters(self, params):
        assert set(params.keys()) == set(TaperedNeuron._defaults.keys())
        # Ignore center
        assert has_positive_values(params,
                                   set(params.keys())-set(('soma_x', 'soma_y', 'soma_z')))

        assert params['soma_rad'] > params['dendh_rad']
        assert params['soma_rad'] > params['axonh_rad']
        
        # TODO: rest

    def contains(self, point, tol):
        '''Is point inside shape?'''
        pass
        # FIXME

    def as_gmsh(self, factory, tag=-1):
        '''Add shape to model in terms of factory(gmsh) primitives'''
        print '>>', self._params
        params = as_namedtuple(self._params)
        
        #^ z-axis          z-Point
        #|                 A2
        #| Axon            A1
        #| Axon hil        A0
        #| Som             Center
        #| Dend hil        D0
        #| Dend            D1
        #|                 D2

        center = np.array([params.soma_x, params.soma_y, params.soma_z])
        # Move up
        shift = sqrt(params.soma_rad**2 - params.axonh_rad**2)
        
        A0 = center + np.array([0, 0, shift])
        A1 = A0 + np.array([0, 0, params.axonh_len])
        A2 = A1 + np.array([0, 0, params.axon_len])

        # Move down
        shift = sqrt(params.soma_rad**2 - params.dendh_rad**2)
        
        D0 = center - np.array([0, 0, shift])
        D1 = D0 - np.array([0, 0, params.dendh_len])
        D2 = D1 - np.array([0, 0, params.dend_len])

        soma = factory.addSphere(params.soma_x, params.soma_y, params.soma_z, params.soma_rad)
        print soma
        
        args = np.r_[A0, A1, params.axonh_rad, params.axon_rad]
        axonh = factory.addCone(*args)
        print axonh
        
        args = np.r_[A1, A2, params.axon_rad]
        axon = factory.addCylinder(*args)
        print axon
        
        print factory.fuse([(3, soma)], [(3, axon), (3, axonh)])
        
# --------------------------------------------------------------------

if __name__ == '__main__':

    import gmsh
    import sys

    model = gmsh.model
    factory = model.occ

    gmsh.initialize(sys.argv)

    gmsh.option.setNumber("General.Terminal", 1)

    neuron = TaperedNeuron()
    
    neuron.as_gmsh(factory)
    factory.synchronize();

    model.mesh.generate(3)
    #model.mesh.refine()
    #model.mesh.setOrder(2)
    #model.mesh.partition(4)
    
    gmsh.write("neuron.msh")
    
    gmsh.finalize()
