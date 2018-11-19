# This example demonstrates the entire pipeline from geometry creation,
# to mesh generation and finally solution. In a realistic workflow
# the 3 steps are done separately.

from mesh.simple_geometry.shapes import (SphereNeuron, MainenNeuron,
                                         FancyProbe, PixelProbe)
from mesh.simple_geometry.geogen import geofile
from mesh.msh_convert import convert

from solver.neuron_solver import neuron_solver
from solver.aux import snap_to_nearest
from solver.aux import load_mesh
from solver.probing import probing_locations, plot_contacts
from dolfin import *
import numpy as np

import subprocess, os
import matplotlib.pyplot as plt


conv = 1E-4

dxp = 200
dxn = 40
dy = 20
dz = 20

scale = 5

geometrical_params = {'rad_soma': 10 * conv*scale, 'rad_dend': 2.5 * conv*scale, 'rad_axon': 1 * conv*scale,
                      'length_dend': 400 * conv, 'length_axon': 200 * conv, 'rad_hilox_d': 4 * conv*scale,
                      'length_hilox_d': 20 * conv, 'rad_hilox_a': 2 * conv*scale, 'length_hilox_a': 10 * conv,
                      'dxp': dxp * conv, 'dxn': dxn * conv, 'dy': dy * conv, 'dz': dz * conv}

geometrical_params.update({'dist': 120*conv})

neuron = MainenNeuron(geometrical_params)

# Mainen - no probe / pixel / fancy
# Sphere - no probe / fancy
probe = FancyProbe({'probe_x': 200*conv, 'probe_y': 0*conv, 'probe_z': -200*conv,
                    'with_contacts': 1})
    
mesh_sizes = {'neuron_mesh_size': 0.1, 'probe_mesh_size': 0.1, 'rest_mesh_size': 0.2}

# This will give us test.GEO
# NOTE: hide_neuron uses geometry of the neuron to get bounding box etc
# but the neuron is not included in the mesh size
geo_file = geofile(neuron, mesh_sizes, probe=probe, hide_neuron=False, file_name='test')
assert os.path.exists('test.GEO')


exit()
# Generate msh file, test.msh
if not os.path.exists('test.h5'):
    subprocess.call(['gmsh -3 -clscale 0.5 test.GEO'], shell=True)
    assert os.path.exists('test.msh')

    # Conversion to h5 file
    convert('test.msh', 'test.h5')
    assert os.path.exists('test.h5')
mesh_path = 'test.h5'

parameters['allow_extrapolation'] = True
conv = 1E-4

problem_params = {'C_m': 1.0,    # uF/um^2
                  'stim_strength': 10.0,             # mS/cm^2
                  'stim_start': 0.01,                # ms
                  'stim_pos': 350*conv, #[0., 0., 350*conv],    # cm
                  'stim_length': 20*conv,            # cm
                  'cond_int': 7.0,                   # mS/cm^2
                  'cond_ext': 3.0,                   # mS/cm^2
                  'I_ion': 0.0,
                  'grounded_bottom_only': False,
                  'Tstop': 5.}                     # ms            
# Spefication of stimulation consists of 2 parts: 
# probe tag to be stimulated and the current to be prescribed. The
# current has the form normal*A amplitude where normal is INWARD (wrt to probe)
# surface normal at the site (assuming the site is flat). This is because we set
# bcs on extracellular and use its outward normal. A = A(time) is okay
problem_params.update({'stimulated_site': 41,  # or higher by convention
                       'site_current': Expression(('A', '0', '0'), degree=0, A=200, t=0)})


solver_params = {'dt_fem': 1E-2, #1E-3,              # ms
                 'dt_ode': 1E-2, #1E-3,               # ms
                 'linear_solver': 'direct'}

mesh, surfaces, volumes, aux_tags = load_mesh(mesh_path)

ax = plot_contacts(surfaces, aux_tags['contact_surfaces'])
# Where are the probes?
plt.show()

# Solver setup
stream = neuron_solver(mesh_path=mesh_path,               # Units assuming mesh lengths specified in cm:
                      problem_parameters=problem_params,                      # ms
                      solver_parameters=solver_params)

# Compute the areas of neuron subdomains for current normalization
# NOTE: on the first yield the stream returns the subdomain function
# marking the subdomains of the neuron surface
neuron_subdomains = next(stream)
# Advance a few times to see if we can solve
for i in range(10):
    _, u, I, _ = next(stream)

File('test_u.pvd') << u

