# This example demonstrates the entire pipeline from geometry creation,
# to mesh generation and finally solution. In a realistic workflow
# the 3 steps are done separately.

from mesh.simple_geometry.shapes import (SphereNeuron, MainenNeuron,
                                         SphereNeurons2, MainenNeurons2,
                                         FancyProbe, PixelProbe)
from mesh.simple_geometry.geogen import geofile
from mesh.msh_convert import convert
from solver import neuron_solver

import subprocess, os


conv = 1E-4

dxp = 200
dxn = 20
dy = 20
dz = 20

scale = 5

geometrical_params = {'rad_soma': 10 * conv*scale, 'rad_dend': 2.5 * conv*scale, 'rad_axon': 1 * conv*scale,
                      'length_dend': 400 * conv, 'length_axon': 200 * conv, 'rad_hilox_d': 4 * conv*scale,
                      'length_hilox_d': 20 * conv, 'rad_hilox_a': 2 * conv*scale, 'length_hilox_a': 10 * conv,
                      'dxp': dxp * conv, 'dxn': dxn * conv, 'dy': dy * conv, 'dz': dz * conv}

geometrical_params.update({'dist': 120*conv})

# Geometry definition
# neuron = SphereNeurons2({'rad_soma': 0.5,
#                          'rad_dend': 0.3, 'length_dend': 1,
#                          'rad_axon': 0.2, 'length_axon': 1,
#                          'dxp': 1.5, 'dxn': 1.25, 'dy': 1.0, 'dz': 0.2,
#                          'dist': 2})


neuron = MainenNeurons2(geometrical_params)
                      
probe = PixelProbe({'probe_x': 200*conv, 'probe_y': 0*conv, 'probe_z': -200*conv,
                    'with_contacts': 1})
    
mesh_sizes = {'neuron_mesh_size': 0.2, 'probe_mesh_size': 0.2, 'rest_mesh_size': 0.4}

# This will give us test.GEO
geo_file = geofile(neuron, mesh_sizes, probe=probe, file_name='test')
# assert os.path.exists('test.GEO')

# Generate msh file, test.msh
# subprocess.call(['gmsh -3 test.GEO'], shell=True)
# assert os.path.exists('test.msh')

# Conversion to h5 file
# convert('test.msh', 'test.h5')
# assert os.path.exists('test.h5')

# Solving
# neuron_solver(mesh_path='test.h5')
