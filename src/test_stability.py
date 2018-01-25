# Run a simulation and see if the membrane models are stable

from mesh.simple_geometry.shapes import SphereNeuron, CylinderProbe
from mesh.simple_geometry.geogen import geofile
from mesh.msh_convert import convert
from solver.neuron_solver import neuron_solver
from dolfin import *

import subprocess, os

# Solver setup
stream = neuron_solver(mesh_path='test.h5',
                       problem_parameters={'C_m': 1.0e-8,
                       'stim_strength': 0.0,
                       'cond_int': 7.0e-4,
                       'cond_ext': 3.0e-4,
                       'I_ion': 0.0,
                       'Tstop': 0.2},
                       solver_parameters={'dt_fem': 1E-3,
                       'dt_ode': 1E-3,
                       'linear_solver': 'direct'})

# The min(u) should be close to -75 (not -74.1)
for t, u in stream:
    print 'At t = %g max(u) = %g min(u) = %g' % (t, u.vector().max(), u.vector().min())

