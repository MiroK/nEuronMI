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
import sys
from glob import glob
# Shut fenics info up
set_log_level(WARNING)

import subprocess, os
import matplotlib.pyplot as plt
import matplotlib as mpl
import MEAutility as mea


def order_recording_sites(sites1, sites2):
    order = []
    for i_s1, s1 in enumerate(sites1):
        distances = [np.linalg.norm(s1 - s2) for s2 in sites2]
        order.append(np.argmin(distances))

    return np.array(order)

source_folder = os.path.abspath(sys.argv[1])
fold_name = source_folder.split('/')[-2] if source_folder[-1] == '/' else source_folder.split('/')[1]

extra_plot = False
run_pc = False
conv = 1E-4
figsize = (18, 14)

probe_map_folder = 'simulations/noneuron_fancy_0_0_-100_coarse_2_box_5_rot_0_rad_0_wprobe/point'

probe_map_folder_original = 'simulations/noneuron_fancy_0_0_-100_coarse_2_box_6_rot_0_rad_0_wprobe/point'

probe_map_mesh = join(probe_map_folder, 'u_ext.h5')
probe_map_elec = join(probe_map_folder, 'elec_dict.npy')
# mesh_path = '/media/terror/code/source/nEuronMI/src/probe_map/results/' \
#             'noneuron_fancy_0_0_-100_coarse_2_box_6_rot_0_rad_0/' \
#             'noneuron_fancy_0_0_-100_coarse_2_box_6_rot_0_rad_0_wprobe.h5'
mesh_path = 'meshes/' \
            'noneuron_fancy_0_0_-100_coarse_2_box_5_rot_0_rad_0/' \
            'noneuron_fancy_0_0_-100_coarse_2_box_5_rot_0_rad_0_wprobe.h5'
load_all_meshes = True

# load neuron
i_mem = np.loadtxt(glob(join(source_folder, 'bas_imem*.txt'))[0])
v_ext_bas = np.loadtxt(glob(join(source_folder, 'bas_vext*.txt'))[0])
v_ext_hybrid = np.loadtxt(glob(join(source_folder, 'v_ext_hybrid*.txt'))[0])
v_ext_corr = np.loadtxt(glob(join(source_folder, 'v_ext_corr*.txt'))[0])
seg_pos = np.loadtxt(glob(join(source_folder, 'seg_pos*.txt'))[0]).T * conv
pos = np.loadtxt(glob(join(source_folder, 'fem_pos*.txt'))[0])
new_pos = []
pos[:, 1] = np.round(pos[:, 1], decimals=5)
pos_back = pos
for u in np.unique(pos[:, 1]):
    for o in pos[pos[:, 1] == u][pos[pos[:, 1] == u][:, 2].argsort()]:
        new_pos.append(o)
pos = np.asarray(new_pos)

elec_dict = np.load(probe_map_elec).item()
info_mea = {'electrode_name': 'nn_emi', 'pos': pos_back, 'center': False}
nn = mea.return_mea(info=info_mea)

# no_mesh = '../results/mainen_fancy_40_0_-100_coarse_2_box_5_noprobe'
# w_mesh = '../results/mainen_fancy_40_0_-100_coarse_2_box_5_wprobe'
# mainen_fancy_40_0_-100_coarse_2_box_5_noprobe
emi_sites = (np.load(join(source_folder, 'sites.npy')) - [40 * conv, 0, 0]) / conv

new_emis = []
emi_sites[:, 1] = np.round(emi_sites[:, 1], decimals=5)
emi_sites_back = emi_sites
for u in np.unique(emi_sites[:, 1]):
    for o in emi_sites[emi_sites[:, 1] == u][emi_sites[emi_sites[:, 1] == u][:, 2].argsort()]:
        new_emis.append(o)
emi_sites = np.asarray(new_emis)

nat_to_back = []
for i, p in enumerate(pos):
    for j, pb in enumerate(pos_back):
        if (pb == p).all():
            nat_to_back.append(j)
nat_to_back = np.asarray(nat_to_back)

no_mesh = '../results/mainen_fancy_40_0_-100_coarse_0_box_3_noprobe'
w_mesh = '../results/mainen_fancy_40_0_-100_coarse_0_box_3_wprobe'

order = order_recording_sites(pos, emi_sites)
order_back = order_recording_sites(pos_back, emi_sites_back)
v_ext_emi_noprobe = np.load(join(source_folder, 'v_ext_emi_noprobe.npy'))[order_back] * 1000
v_ext_emi_wprobe = np.load(join(source_folder, 'v_ext_emi_wprobe.npy'))[order_back] * 1000
# v_ext_emi_noprobe = np.load(join(source_folder, 'v_ext_emi_noprobe.npy'))[order] * 1000
# v_ext_emi_wprobe = np.load(join(source_folder, 'v_ext_emi_wprobe.npy'))[order] * 1000

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

    np.savetxt('v_ext_corr.txt', v_ext_corr)

# scaling = np.max(np.abs(v_ext_bas))/np.max(np.abs(v_ext_corr))
# v_ext = np.array([v_ext_corr, v_ext_bas, v_ext_bas*1.8])
v_ext = np.array([v_ext_hybrid, v_ext_bas, v_ext_bas * 2, v_ext_corr])
v_emi_hybrid = np.array([v_ext_hybrid, v_ext_emi_wprobe])
v_bas_emi_noprobe = np.array([v_ext_bas, v_ext_emi_noprobe])
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_bas = colors[3]
color_hybrid = colors[2]
color_moi = colors[6]
color_18moi = colors[7]
color_emi = colors[1]
color_emi_noprobe = colors[0]
end_T = 5.
# mea.plot_mea_recording(v_ext_corr, nn, scalebar=True)
# mea.plot_mea_recording(v_ext_bas, nn)
# mea.plot_mea_recording(v_ext_hybrid, nn)


fig1 = plt.figure(figsize=figsize)
ax1 = fig1.add_subplot(1,2,1)
mea.plot_mea_recording(np.array([v_ext_bas, v_ext_emi_noprobe]), nn, colors=[color_bas, color_emi_noprobe],
                       lw=1.5, ax=ax1, scalebar=True, time=end_T, vscale=40)
ax1.legend(labels=['CE', 'EMI (no probe)'], fontsize=18, loc='upper right', ncol=1)

ax2 = fig1.add_subplot(1,2,2)
mea.plot_mea_recording(np.array([v_ext_hybrid, v_ext_emi_wprobe]), nn,
                       colors=[color_hybrid, color_emi], lw=1.5, ax=ax2, scalebar=True, time=end_T, vscale=40)
ax2.legend(labels=['HS', 'EMI (with probe)'], fontsize=18, loc='upper right', ncol=1)

simplify_axes([ax1, ax2])
mark_subplots([ax1, ax2], xpos=-0.1, ypos=1.02, fs=45)

# plt.tight_layout()
# plt.figure()
fig2 = plt.figure(figsize=figsize)
ax21 = fig2.add_subplot(1,2,1)
ax21 = mea.plot_mea_recording(np.array([2 * v_ext_bas, 1.8 * v_ext_bas, v_ext_hybrid]), nn,
                             colors=[color_moi, color_18moi, color_hybrid], ax=ax21,
                             lw=1.5, scalebar=True, time=end_T, vscale=40)
ax21.legend(labels=['MoI', '1.8 MoI','HS'], fontsize=18, loc='upper right', ncol=3)

# ratios
ratio_bas_hyb = np.max(np.abs(v_ext_bas[:,150:350]), axis=1) / np.max(np.abs(v_ext_hybrid[:,150:350]), axis=1)
ratio_moi_hyb = np.max(np.abs(2 * v_ext_bas[:,150:350]), axis=1) / np.max(np.abs(v_ext_hybrid[:,150:350]), axis=1)
ratio_moi18_hyb = np.max(np.abs(1.8 * v_ext_bas[:,150:350]), axis=1) / np.max(np.abs(v_ext_hybrid[:,150:350]), axis=1)
ratio_corr_hyb = np.max(np.abs(v_ext_corr[:,150:350]), axis=1) / np.max(np.abs(v_ext_hybrid[:,150:350]), axis=1)
ratio_corr_emi = np.max(np.abs(v_ext_corr[:,150:350]), axis=1) / np.max(np.abs(v_ext_emi_wprobe[:,150:350]), axis=1)
ratio_bas_emi = np.max(np.abs(v_ext_bas[:,150:350]), axis=1) / np.max(np.abs(v_ext_emi_wprobe[:,150:350]), axis=1)
ratio_moi18_emi = np.max(np.abs(1.8 * v_ext_bas[:,150:350]), axis=1) / np.max(np.abs(v_ext_emi_wprobe[:,150:350]), axis=1)
ratio_moi_emi = np.max(np.abs(2 * v_ext_bas[:,150:350]), axis=1) / np.max(np.abs(v_ext_emi_wprobe[:,150:350]), axis=1)

ratio_hyb_hyb = np.ones(2)

ax22 = fig2.add_subplot(1,2,2)
sns.distplot(ratio_bas_hyb, bins=20, hist=False, rug=True, color=color_bas , label='CE', ax=ax22)
sns.distplot(ratio_moi_hyb, bins=20, hist=False, rug=True, color=color_moi, label='MoI', ax=ax22)
sns.distplot(ratio_moi18_hyb, bins=20, hist=False, rug=True, color=color_18moi, label='1.8MoI', ax=ax22)
# sns.distplot(ratio_hyb_hyb, bins=20, hist=False, rug=True, color=color_hybrid, label='HS', ax=ax22)
ax22.vlines(1., ymin=0, ymax=10, colors=color_hybrid, label='PC', linewidth=2)
ax22.set_title('Peak ratio with HS', fontsize=22)
ax22.set_xlabel('Peak ratio', fontsize=15)
ax22.set_ylabel('Frequency', fontsize=15)
ax22.legend(fontsize=18, loc='upper right')

# ax11 = fig.add_subplot(1, 2, 2)
# sns.distplot(ratio_bas_emi, bins=20, hist=False, rug=True, label='BAS', ax=ax11)
# sns.distplot(ratio_corr_emi, bins=20, hist=False, rug=True, label='PC', ax=ax11)
# sns.distplot(ratio_moi_emi, bins=20, hist=False, rug=True, label='MoI', ax=ax11)
# sns.distplot(ratio_moi18_emi, bins=20, hist=False, rug=True, label='1.8MoI', ax=ax11)
# ax11.set_title('Peak ratio with EMI')
# ax11.legend(fontsize=18, loc='upper right')

simplify_axes([ax22])
mark_subplots([ax21, ax22], xpos=-0.1, ypos=1.02, fs=45)

fig1.subplots_adjust(left=0.1, bottom=0.05, right=0.95, top=0.9, wspace=0.2)
fig2.subplots_adjust(left=0.1, bottom=0.05, right=0.95, top=0.9, wspace=0.2)

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
print 'peak MoI: ', np.min(2 * v_ext_bas)
print 'difference wprobe: ', np.min(v_ext_emi_wprobe) - np.min(2 * v_ext_bas)

if extra_plot:
    plt.figure(figsize=[15, 10])
    plt.suptitle('Paired comparison left Vs right electrodes, intra modality [{}]'.format(fold_name))
    ax_1 = plt.subplot(121)

    # plt.title('BAS and Hybrid')
    norm = mpl.colors.Normalize(0, 9)
    # col = (cell.vmem.T[spike_time_loc[0]] + 100) / 150.
    # col = {'soma': 'k', 'axon': 'b', 'dendrite': 'r', }
    # norm = mpl.colors.Normalize(0, n_sec)
    set2 = plt.get_cmap('tab10')
    colr = plt.cm.ScalarMappable(norm=norm, cmap=set2)
    for i in range(10):
        ax_1.plot(v_ext_bas[nat_to_back[i]], color=colr.to_rgba(i), linestyle='-', label='electrode {}'.format(i))
        ax_1.plot(v_ext_bas[nat_to_back[i + 22]], color=colr.to_rgba(i), linestyle='--', label='electrode {}'.format(i + 22))
    ax_1.legend()
    ax_1.set_title('BAS')

    ax_2 = plt.subplot(122)
    # plt.title('Hybrid')
    norm = mpl.colors.Normalize(0, 9)
    # col = (cell.vmem.T[spike_time_loc[0]] + 100) / 150.
    # col = {'soma': 'k', 'axon': 'b', 'dendrite': 'r', }
    # norm = mpl.colors.Normalize(0, n_sec)
    set2 = plt.get_cmap('tab10')
    colr = plt.cm.ScalarMappable(norm=norm, cmap=set2)
    for i in range(10):
        ax_2.plot(v_ext_hybrid[nat_to_back[i]], color=colr.to_rgba(i), linestyle='-', label='electrode {}'.format(i))
        ax_2.plot(v_ext_hybrid[nat_to_back[i + 22]], color=colr.to_rgba(i), linestyle='--', label='electrode {}'.format(i + 22))
    # ax_2.legend()
    ax_2.set_title('Hybrid')


    ymin = np.min([np.min(v_ext_bas), np.min(v_ext_emi_wprobe)])
    ymax = np.max([np.max(v_ext_bas), np.max(v_ext_emi_wprobe)])
    plt.figure(figsize=[20, 20])
    plt.suptitle('Paired comparison of largest inter modality differences per electrode [{}]'.format(fold_name))
    ax_a = plt.subplot(221)
    n_elec = 4
    norm = mpl.colors.Normalize(0, n_elec)
    set2 = plt.get_cmap('tab10')
    colr = plt.cm.ScalarMappable(norm=norm, cmap=set2)
    ax_a.set_title('BAS - EMI')
    temp = ratio_moi_emi.argsort()
    for ix, i in enumerate(temp[-n_elec:]):
        natural_electrode = np.where(nat_to_back == i)[0][0]
        ax_a.plot(v_ext_bas[i], color=colr.to_rgba(ix), linestyle='-', label='electrode {}'.format(natural_electrode))
        ax_a.plot(v_ext_emi_wprobe[i], color=colr.to_rgba(ix), linestyle='--', label='electrode {}'.format(natural_electrode))  # emi same electrode data
        ax_a.plot(v_ext_bas[i] - v_ext_emi_wprobe[i], color=colr.to_rgba(ix), linestyle=':', label='difference')
    # ratio_moi18_emi
    ax_a.legend()


    ax_b = plt.subplot(222)
    n_elec = 4
    norm = mpl.colors.Normalize(0, n_elec)
    set2 = plt.get_cmap('tab10')
    colr = plt.cm.ScalarMappable(norm=norm, cmap=set2)
    ax_b.set_title('MOI - EMI')
    temp = ratio_bas_emi.argsort()
    for ix, i in enumerate(temp[-n_elec:]):
        natural_electrode = np.where(nat_to_back == i)[0][0]
        ax_b.plot(2 * v_ext_bas[i], color=colr.to_rgba(ix), linestyle='-', label='electrode {}'.format(natural_electrode))
        ax_b.plot(v_ext_emi_wprobe[i], color=colr.to_rgba(ix), linestyle='--', label='electrode {}'.format(natural_electrode))  # emi same electrode data
        ax_b.plot(2 * v_ext_bas[i] - v_ext_emi_wprobe[i], color=colr.to_rgba(ix), linestyle=':', label='difference')
    # ratio_moi18_emi
    ax_b.legend()
    ymin_t = np.min([np.min(2 * v_ext_bas), np.min(v_ext_emi_wprobe)])
    ymax_t = np.max([np.max(2 * v_ext_bas), np.max(v_ext_emi_wprobe)])
    if ymin_t < ymin:
        ymin = ymin_t
    if ymax_t > ymax:
        ymax = ymax_t



    ax_c = plt.subplot(223)
    n_elec = 4
    norm = mpl.colors.Normalize(0, n_elec)
    set2 = plt.get_cmap('tab10')
    colr = plt.cm.ScalarMappable(norm=norm, cmap=set2)
    ax_c.set_title('Corr - Hybrid')
    temp = ratio_corr_hyb.argsort()
    for ix, i in enumerate(temp[-n_elec:]):
        natural_electrode = np.where(nat_to_back == i)[0][0]
        ax_c.plot(v_ext_corr[i], color=colr.to_rgba(ix), linestyle='-', label='electrode {}'.format(natural_electrode))
        ax_c.plot(v_ext_hybrid[i], color=colr.to_rgba(ix), linestyle='--', label='electrode {}'.format(natural_electrode))  # emi same electrode data
        ax_c.plot(v_ext_corr[i] - v_ext_hybrid[i], color=colr.to_rgba(ix), linestyle=':', label='difference')
    # ratio_moi18_emi
    ax_c.legend()
    ymin_t = np.min([np.min(v_ext_corr), np.min(v_ext_hybrid)])
    ymax_t = np.max([np.max(v_ext_corr), np.max(v_ext_hybrid)])
    if ymin_t < ymin:
        ymin = ymin_t
    if ymax_t > ymax:
        ymax = ymax_t

    ax_d = plt.subplot(224)
    n_elec = 4
    norm = mpl.colors.Normalize(0, n_elec)
    set2 = plt.get_cmap('tab10')
    colr = plt.cm.ScalarMappable(norm=norm, cmap=set2)
    ax_d.set_title('PC - EMI')
    temp = ratio_corr_emi.argsort()
    for ix, i in enumerate(temp[-n_elec:]):
        natural_electrode = np.where(nat_to_back == i)[0][0]
        ax_d.plot(v_ext_corr[i], color=colr.to_rgba(ix), linestyle='-', label='electrode {}'.format(natural_electrode))
        ax_d.plot(v_ext_emi_wprobe[i], color=colr.to_rgba(ix), linestyle='--', label='electrode {}'.format(natural_electrode))  # emi same electrode data
        ax_d.plot(v_ext_corr[i] - v_ext_emi_wprobe[i], color=colr.to_rgba(ix), linestyle=':', label='difference')
    # ratio_moi18_emi
    ax_d.legend()
    ax_d.set_ylim([ymin, ymax])
    ax_c.set_ylim([ymin, ymax])
    ax_b.set_ylim([ymin, ymax])
    ax_a.set_ylim([ymin, ymax])
