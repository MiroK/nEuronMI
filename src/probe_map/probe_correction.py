# This example demonstrates the entire pipeline from geometry creation,
# to mesh generation and finally solution. In a realistic workflow
# the 3 steps are done separately.


import sys

sys.path.append('../solver')
from aux import load_mesh
from probing import probing_locations, plot_contacts, probe_contact_map
from dolfin import *
import numpy as np
from os.path import join

# Shut fenics info up
set_log_level(WARNING)

import subprocess, os
import matplotlib.pyplot as plt
import MEAutility as mea

def monopolar(I, elec, pos, den=4):
    '''

    Parameters
    ----------
    I
    elec
    pos
    den

    Returns
    -------

    '''
    return np.array([I*1e-4 / (den*np.pi*0.3*np.linalg.norm(elec - p)) for p in pos])

conv = 1E-4
probe_map_folder = '/home/alessio/Documents/Codes/nEuronMI/src/probe_map/simulations/' \
                   'noneuron_fancy_0_0_-100_coarse_2_box_6_rot_0_rad_0_wprobe/distr'
probe_map_mesh = join(probe_map_folder, 'u_ext.h5')
probe_map_elec = join(probe_map_folder, 'elec_dict.npy')
mesh_path = '/home/alessio/Documents/Codes/nEuronMI/src/probe_map/meshes/' \
            'noneuron_fancy_0_0_-100_coarse_2_box_6_rot_0_rad_0/' \
            'noneuron_fancy_0_0_-100_coarse_2_box_6_rot_0_rad_0_wprobe.h5'
mesh, surfaces, volumes, aux_tags = load_mesh(mesh_path)

# load neuron
i_mem = np.loadtxt('bas_imem.txt')
v_ext_bas = np.loadtxt('bas_vext.txt')
seg_pos = np.loadtxt('seg_pos.txt').T * conv
pos = np.loadtxt('fem_pos.txt')
v_ext_corr = np.zeros((len(pos), i_mem.shape[1]))

elec_dict = np.load(probe_map_elec).item()
info_mea = {'electrode_name': 'nn_emi', 'pos': pos, 'center': False}
nn = mea.return_mea(info=info_mea)

# Read back from file later
hdf5_file = HDF5File(mesh.mpi_comm(), probe_map_mesh, "r")
dist = []
gain = []
for i, (site, elec) in enumerate(elec_dict.items()):
    single_dist = []
    single_gain = []
    # Recreate the function space on mesh and fill the vector
    V = FunctionSpace(mesh, 'CG', 1)
    f = Function(V)
    hdf5_file.read(f, '/function_%d' % site)

    # Finally this is how to do shifted potentials
    f_shift = lambda x, f=f, c=elec: f(x - c)
    print 'Computing currents for electrode: ', i + 1, ' (site ', site, ')'

    for seg, im in zip(seg_pos, i_mem):
        v_ext_corr[i] += im * f(seg)
        single_dist.append(np.linalg.norm(elec - seg))
        single_gain.append(f(seg))

    dist.append(single_dist)
    gain.append(single_gain)

dist = np.array(dist)
gain = np.array(gain)

scaling = np.max(np.abs(v_ext_bas))/np.max(np.abs(v_ext_corr))
v_ext = np.array([-v_ext_corr*scaling, v_ext_bas])

mea.plot_mea_recording(-v_ext_corr, nn, scalebar=True)
mea.plot_mea_recording(v_ext_bas, nn)
mea.plot_mea_recording(v_ext, nn)

# decay with distance:
x = np.linspace(-0.03, -0.00075, 1000)
y = np.ones(1000) * elec[1]
z = np.ones(1000) * elec[2]

dec_pos = np.array([x,y,z]).T
v_dec = np.array([f(d) for d in dec_pos])

dist_line = [np.linalg.norm(d-elec) for d in dec_pos]

v_mono4 = monopolar(-1, elec, dec_pos)
v_mono2 = monopolar(-1, elec, dec_pos, den=2)
v_mono3 = monopolar(-1, elec, dec_pos, den=3)

