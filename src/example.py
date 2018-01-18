# This example demonstrates the entire pipeline from geometry creation,
# to mesh generation and finally solution. In a realistic workflow
# the 3 steps are done separately.

from mesh.simple_geometry.shapes import SphereNeuron, CylinderProbe
from mesh.simple_geometry.geogen import geofile
from mesh.msh_convert import convert
from solver.neuron_solver import neuron_solver

import subprocess, os

h5_is_done = False   # Set to true when the test.h5 is already in place

# Geometry definition
neuron = SphereNeuron({'rad_soma': 0.5,
                       'rad_dend': 0.3, 'length_dend': 1,
                       'rad_axon': 0.2, 'length_axon': 1,
                       'dxp': 1.5, 'dxn': 1.25, 'dy': 1.0, 'dz': 0.2})

probe = CylinderProbe({'rad_probe': 0.2, 'probe_x': 1.5, 'probe_y': 0, 'probe_z': 0})
    
mesh_sizes = {'neuron_mesh_size': 0.2, 'probe_mesh_size': 0.2, 'rest_mesh_size': 0.4}

# This will give us test.GEO
# geo_file = geofile(neuron, mesh_sizes, probe=probe, file_name='test')
assert h5_is_done or os.path.exists('test.GEO')

# Generate msh file, test.msh
# subprocess.call(['gmsh -3 test.GEO'], shell=True)
assert h5_is_done or os.path.exists('test.msh')

# Conversion to h5 file
# convert('test.msh', 'test.h5')
assert h5_is_done or os.path.exists('test.h5')

# Solving
neuron_solver(mesh_path='test.h5',
              problem_parameters={'C_m': 1E-3,
                                  'cond_int': 1.0,
                                  'cond_ext': 1.2,
                                  'I_ion': 0.0,
                                  'Tstop': 1.0},
              solver_parameters={'dt_fem': 1E-3,
                                 'dt_ode': 1E-4,
                                 'linear_solver': 'direct'})
