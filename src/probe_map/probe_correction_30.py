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
import time
import seaborn as sns
from neuroplot import *

# Shut fenics info up
set_log_level(WARNING)

import subprocess, os
import matplotlib.pyplot as plt
import MEAutility as mea

def order_recording_sites(sites1, sites2):
    order = []
    for i_s1, s1 in enumerate(sites1):
        distances = [np.linalg.norm(s1 - s2) for s2 in sites2]
        order.append(np.argmin(distances))

    return np.array(order)

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
    return np.array([I / (den*np.pi*0.3*np.linalg.norm(elec - p)*1e4) for p in pos]) * 1000


run_pc = False
conv = 1E-4

probe_map_folder = '/home/alessio/Documents/Codes/nEuronMI/src/probe_map/simulations/' \
                   'noneuron_fancy_0_0_-100_coarse_2_box_5_rot_0_rad_0_wprobe/point'
probe_map_mesh = join(probe_map_folder, 'u_ext.h5')
probe_map_elec = join(probe_map_folder, 'elec_dict.npy')
mesh_path = '/home/alessio/Documents/Codes/nEuronMI/src/probe_map/meshes/' \
            'noneuron_fancy_0_0_-100_coarse_2_box_5_rot_0_rad_0/' \
            'noneuron_fancy_0_0_-100_coarse_2_box_5_rot_0_rad_0_wprobe.h5'

load_all_meshes = True

# load neuron
i_mem = np.loadtxt('bas_imem_30.txt')
v_ext_bas = np.loadtxt('bas_vext_30.txt')
v_ext_hybrid = np.loadtxt('v_ext_hybrid_30.txt')
if os.path.isfile('v_ext_corr_30.txt'):
    v_ext_corr = np.loadtxt('v_ext_corr_30.txt')
else:
    run_pc=True
seg_pos = np.loadtxt('seg_pos_30.txt').T * conv
pos = np.loadtxt('fem_pos.txt')


elec_dict = np.load(probe_map_elec).item()
info_mea = {'electrode_name': 'nn_emi', 'pos': pos, 'center': False}
nn = mea.return_mea(info=info_mea)

no_mesh = '../results/mainen_fancy_40.0_30.0_-100.0_coarse_2_box_5_noprobe'
w_mesh = '../results/mainen_fancy_40.0_30.0_-100.0_coarse_2_box_5_wprobe'

emi_sites = (np.load(join(no_mesh, 'sites.npy')) - [40*conv, 0, 0]) / conv
# center emi_sites
emi_sites_c = emi_sites - np.mean(emi_sites, axis=0)
z_shift = np.max(emi_sites_c[:, 2]) - np.max(pos[:, 2])
emi_sites_c = emi_sites_c - [0,0,z_shift]
order = order_recording_sites(pos, emi_sites_c)
v_ext_emi_noprobe = np.load(join(no_mesh, 'v_probe.npy'))[order]*1000
v_ext_emi_wprobe = np.load(join(w_mesh, 'v_probe.npy'))[order]*1000
info_mea = {'electrode_name': 'nn_emi', 'pos': emi_sites, 'center': False}
emi_nn = mea.return_mea(info=info_mea)

elec = elec_dict[61]

# Read back from file later
if run_pc:
    v_ext_corr = np.zeros((len(pos), i_mem.shape[1]))
    mesh, surfaces, volumes, aux_tags = load_mesh(mesh_path)

    hdf5_file = HDF5File(mesh.mpi_comm(), probe_map_mesh, "r")
    dist = []
    gain = []
    if load_all_meshes:
        functions = []
        for i, (site, elec) in enumerate(elec_dict.items()):
            print 'Loading solution: ', i + 1, ' of ', len(nn.positions)
            V = FunctionSpace(mesh, 'CG', 1)
            f = Function(V)
            hdf5_file.read(f, '/function_%d' % site)
            functions.append(f)

    t_start = time.time()
    for i, (site, elec) in enumerate(elec_dict.items()):
        single_dist = []
        single_gain = []
        # Recreate the function space on mesh and fill the vector
        if not load_all_meshes:
            V = FunctionSpace(mesh, 'CG', 1)
            f = Function(V)
            hdf5_file.read(f, '/function_%d' % site)
        else:
            f = functions[i]

        # Finally this is how to do shifted potentials
        print 'Computing currents for electrode: ', i + 1, ' (site ', site, ')'

        for seg, im in zip(seg_pos, i_mem):
            v_ext_corr[i] += im * f(seg)
            single_dist.append(np.linalg.norm(elec - seg))
            single_gain.append(f(seg))
        print 'Elapsed time: ', time.time() - t_start

        dist.append(single_dist)
        gain.append(single_gain)

    dist = np.array(dist)
    gain = np.array(gain)
    v_ext_corr *= 1000

    np.savetxt('v_ext_corr_30.txt', v_ext_corr*1000)

# scaling = np.max(np.abs(v_ext_bas))/np.max(np.abs(v_ext_corr))
# v_ext = np.array([v_ext_corr, v_ext_bas, v_ext_bas*1.5])
v_ext = np.array([v_ext_hybrid, v_ext_bas, v_ext_bas*2, v_ext_corr])
v_emi_hybrid = np.array([v_ext_hybrid, v_ext_emi_wprobe])
v_bas_emi_noprobe = np.array([v_ext_bas, v_ext_emi_noprobe])


# mea.plot_mea_recording(v_ext_corr, nn, scalebar=True)
# mea.plot_mea_recording(v_ext_bas, nn)
# mea.plot_mea_recording(v_ext_hybrid, nn)
ax1 = mea.plot_mea_recording(v_ext, nn, lw=1.5, vscale=40)
ax1.legend(labels=['HS', 'BAS', 'MoI', 'PC'], fontsize=18, loc='upper right', ncol=4)

ax2 = mea.plot_mea_recording(v_emi_hybrid, nn, lw=1.5, vscale=40)
ax2.legend(labels=['HS', 'EMI'], fontsize=18, loc='upper right', ncol=2)

ax3 = mea.plot_mea_recording(v_bas_emi_noprobe, nn, lw=1.5, vscale=40)
ax3.legend(labels=['BAS', 'EMI no probe'], fontsize=18, loc='upper right', ncol=2)


# # decay with distance:
# x = np.linspace(-0.03, -0.00075, 1000)
# y = np.ones(1000) * elec[1]
# z = np.ones(1000) * elec[2]
#
# dec_pos = np.array([x,y,z]).T
# v_dec = np.array([f(d) for d in dec_pos])
#
# dist_line = [np.linalg.norm(d-elec) for d in dec_pos]
#
# v_mono4 = monopolar(1, elec, dec_pos)
# v_mono2 = monopolar(1, elec, dec_pos, den=2)
# v_mono3 = monopolar(1, elec, dec_pos, den=3)
#
# plt.figure()
# plt.semilogy(dist_line, v_dec)
# plt.semilogy(dist_line, v_mono4)
# plt.semilogy(dist_line, v_mono2)

# y_probe = np.linspace(-0.01, 0.02, 100)
# z_probe = np.linspace(-0.01, 0.02, 100)
# v_probe = np.zeros((1000,1000))
# for i, yp in enumerate(y_probe):
#     for j, zp in enumerate(z_probe):
#         v_probe[i,j] = f([-7.5*conv, yp, zp])
# plt.matshow(v_probe)

# ratios
ratio_bas_hyb = np.max(np.abs(v_ext_bas), axis=1) / np.max(np.abs(v_ext_hybrid), axis=1)
ratio_moi_hyb = np.max(np.abs(2*v_ext_bas), axis=1) / np.max(np.abs(v_ext_hybrid), axis=1)
ratio_moi18_hyb = np.max(np.abs(1.6*v_ext_bas), axis=1) / np.max(np.abs(v_ext_hybrid), axis=1)
ratio_corr_hyb = np.max(np.abs(v_ext_corr), axis=1) / np.max(np.abs(v_ext_hybrid), axis=1)
ratio_corr_emi = np.max(np.abs(v_ext_corr), axis=1) / np.max(np.abs(v_ext_emi_wprobe), axis=1)
ratio_bas_emi = np.max(np.abs(v_ext_bas), axis=1) / np.max(np.abs(v_ext_emi_wprobe), axis=1)
ratio_moi18_emi = np.max(np.abs(1.6*v_ext_bas), axis=1) / np.max(np.abs(v_ext_emi_wprobe), axis=1)
ratio_moi_emi = np.max(np.abs(2*v_ext_bas), axis=1) / np.max(np.abs(v_ext_emi_wprobe), axis=1)
fig = plt.figure()

ax22 = fig.add_subplot(1,2,1)
sns.distplot(ratio_bas_hyb, bins=20, hist=False, rug=True, label='BAS', ax=ax22)
sns.distplot(ratio_moi_hyb, bins=20, hist=False, rug=True, label='MoI', ax=ax22)
sns.distplot(ratio_moi18_hyb, bins=20, hist=False, rug=True, label='1.6MoI', ax=ax22)
ax22.set_title('Peak ratio with  hybrid')
ax22.legend(fontsize=18, loc='upper right')

ax11 = fig.add_subplot(1,2,2)
sns.distplot(ratio_bas_emi, bins=20, hist=False, rug=True, label='BAS', ax=ax11)
sns.distplot(ratio_corr_emi, bins=20, hist=False, rug=True, label='PC', ax=ax11)
sns.distplot(ratio_moi_emi, bins=20, hist=False, rug=True, label='MoI', ax=ax11)
sns.distplot(ratio_moi18_emi, bins=20, hist=False, rug=True, label='1.6MoI', ax=ax11)
ax11.set_title('Peak ratio with EMI')
ax11.legend(fontsize=18, loc='upper right')

simplify_axes([ax11, ax22])
mark_subplots([ax22, ax11], xpos=-0.1, ypos=1.02, fs=35)

# sns.distplot(ratio_corr_hyb, bins=10)

print 'NEURON min at: ', np.unravel_index(v_ext_bas.argmin(), v_ext_bas.shape)
print 'EMI min at: ', np.unravel_index(v_ext_emi_noprobe.argmin(), v_ext_emi_noprobe.shape)
print '\n'
print 'peak NEURON: ', np.min(v_ext_bas)
print 'peak EMI noprobe: ', np.min(v_ext_emi_noprobe)
print 'difference noprobe: ', np.min(v_ext_emi_noprobe) - np.min(v_ext_bas)
print 'peak EMI wprobe: ', np.min(v_ext_emi_wprobe)
print 'difference wprobe: ', np.min(v_ext_emi_wprobe) - np.min(v_ext_bas)
print 'peak HYBRID: ', np.min(v_ext_hybrid)
print 'difference wprobe: ', np.min(v_ext_emi_wprobe) - np.min(v_ext_hybrid)
print 'peak MoI: ', np.min(2*v_ext_bas)
print 'difference wprobe: ', np.min(v_ext_emi_wprobe) - np.min(2*v_ext_bas)

