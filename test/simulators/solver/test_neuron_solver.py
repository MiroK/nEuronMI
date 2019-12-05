from neuronmi.simulators.solver.neuron_solver import neuron_solver
from neuronmi.mesh.mesh_utils import EMIEntityMap

import itertools, unittest, os, subprocess
import numpy as np

from dolfin import *


class TestEMI(unittest.TestCase):
    mesh_path = './sandbox/test_2neuron.h5'

    @classmethod
    def setUpClass(cls):
        not os.path.exists(TestEMI.mesh_path) and subprocess.call(['python two_neurons.py'], cwd='./sandbox', shell=True)

    # Current magnitude for probe tip
    magnitude = Expression('exp(-1E-2*t)', t=0, degree=1)
        
    problem_parameters = {'neuron_0': {'I_ion': Constant(0),
                                       'cond': 1,
                                       'C_m': 1,
                                       'stim_strength': 0.0,
                                       'stim_start': 0.0,  
                                       'stim_pos': 0.0,
                                       'stim_length': 0.0},
                          #
                          'neuron_1': {'I_ion': Constant(0),
                                       'cond': 1,
                                       'C_m': 1,
                                       'stim_strength': 0.0,
                                       'stim_start': 0.0,  
                                       'stim_pos': 0.0,
                                       'stim_length': 0.0},
                          #
                          'external': {'cond': 2,
                          # Box boundaries where zero current should be
                          # precscribed. If nono, all surface are grounded
                                       'insulated_bcs': ('max_x', 'max_y'),},
                          #
                          'probe': {'stimulated_sites': ('contact_0', ),
                                    'site_currents': (magnitude, )}
                          }
                          
    solver_parameters = {'dt_fem': 0.1,
                         'dt_ode': 0.01,
                         'Tstop': 1}

    def test_emi(self):
        emi_map = './sandbox/test_2neuron.json'
        with open(emi_map) as json_fp:
            emi_map = EMIEntityMap(json_fp=json_fp)

    
        for (t, u, I) in neuron_solver(TestEMI.mesh_path,
                                       emi_map,
                                       TestEMI.problem_parameters,
                                       TestEMI.solver_parameters):
            # I just check that nothing blows up
            self.assertTrue(not np.any(np.isnan(u.vector().get_local())))
            self.assertTrue(not np.any(np.isnan(I.vector().get_local())))

# ---------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
