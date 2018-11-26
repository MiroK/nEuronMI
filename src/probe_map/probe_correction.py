# This example demonstrates the entire pipeline from geometry creation,
# to mesh generation and finally solution. In a realistic workflow
# the 3 steps are done separately.

# from mesh.simple_geometry.shapes import (SphereNeuron, MainenNeuron,
#                                          FancyProbe, PixelProbe)
# from mesh.simple_geometry.geogen import geofile
# from mesh.msh_convert import convert

# from solver.neuron_solver import neuron_solver
# from solver.aux import snap_to_nearest
import sys
sys.path.append('../solver')
from aux import load_mesh
# from solver.probing import probing_locations, plot_contacts, probe_contact_map
from dolfin import *
import numpy as np
from os.path import join

# Shut fenics info up
set_log_level(WARNING)

import subprocess, os
import matplotlib.pyplot as plt

conv = 1E-4
probe_map_folder = '/media/terror/code/source/nEuronMI/src/probe_map/results/noneuron_fancy_0_0_-100_coarse_2_box_6_rot_0_rad_0_wprobe/point'
probe_map_mesh = join(probe_map_folder, 'u_ext.h5')
probe_map_elec = join(probe_map_folder, 'elec_dict.npy')
mesh_path = '/media/terror/code/source/nEuronMI/src/probe_map/results/noneuron_fancy_0_0_-100_coarse_2_box_6_rot_0_rad_0/noneuron_fancy_0_0_-100_coarse_2_box_6_rot_0_rad_0_wprobe.h5'
mesh, surfaces, volumes, aux_tags = load_mesh(mesh_path)


elec_dict = np.load(probe_map_elec).item()
# Read back from file later
hdf5_file = HDF5File(mesh.mpi_comm(), probe_map_mesh, "r")
for site, elec in elec_dict.items():
  # Recreate the function space on mesh and fill the vector
  V = FunctionSpace(mesh, 'CG', 1)
  f = Function(V)
  hdf5_file.read(f, '/function_%d' % site)

  # Finally this is how to do shifted potentials
  f_shift = lambda x, f=f, c=elec: f(x-c)
  print(elec)

  print(f_shift([30 * conv, 0, 0]))

