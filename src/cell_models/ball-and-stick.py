import numpy as np
import matplotlib.pylab as plt
import LFPy
import neuron
import yaml
import MEAutility as mea
# from neuroplot import *
import time
from os.path import join

plt.ion()
plt.show()

def order_recording_sites(sites1, sites2):
    order = []
    for i_s1, s1 in enumerate(sites1):
        distances = [np.linalg.norm(s1 - s2) for s2 in sites2]
        order.append(np.argmin(distances))

    return np.array(order)

t_start = time.time()
# TODO add sort order

conv=1E-4
fs_legend = 20
save_fig = False
figsize = (9, 14)
end_T = 5.

# Load sites
### Compare with EMI ###
no_mesh = '../results/mainen_fancy_40_0_-100_coarse_0_box_3_noprobe'
w_mesh = '../results/mainen_fancy_40_0_-100_coarse_0_box_3_wprobe'
# no_mesh = '../results/mainen_fancy_40_0_-100_coarse_2_box_5_noprobe'
# w_mesh =  '../results/mainen_fancy_40_0_-100_coarse_2_box_5_wprobe'

with open(join(no_mesh, 'params.yaml'), 'r') as f:
    info = yaml.load(f)

T = info['problem']['Tstop']

times = np.load(join(no_mesh, 'times.npy'))
sites = np.load(join(no_mesh, 'sites.npy'))/conv
i_soma = np.load(join(no_mesh, 'i_soma.npy'))


# Define cell parameters
cell_parameters = {
    'morphology' : 'ball_and_stick_waxon_tapered.hoc', # from Mainen & Sejnowski, J Comput Neurosci, 1996
    'cm' : 1.0,         # membrane capacitance
    'Ra' : 150.,        # axial resistance
    'v_init' : -75.,    # initial crossmembrane potential
    'passive' : True,   # turn on NEURONs passive mechanism for all sections
    # 'passive_parameters' : {'g_pas' : 1./30000, 'e_pas' : -75},
    'passive_parameters': {'g_pas': 0.06*1E-3, 'e_pas': -75},
    # 'nsegs_method': 'lambda100',  # spatial discretization method
    'nsegs_method' : 'fixed_length', # spatial discretization method
    'max_nsegs_length' : 1,
    'lambda_f' : 1000.,           # frequency where length constants are computed
    'dt' : 1E-2,      # simulation time step size
    'tstart' : 0.,      # start time of simulation, recorders start at t=0
    'tstop' : end_T,     # stop simulation at 100 ms.
}

# Create cell
cell = LFPy.Cell(**cell_parameters)
for sec in neuron.h.soma:
    sec.insert('hh')
for sec in neuron.h.axon:
    sec.insert('hh')
#
# for sec in neuron.h.allsec():
#     sec.insert('extracellular')
# #
# for sec in neuron.h.allsec():
#     for seg in sec:
#         seg.xg[0] = 0
#         seg.xraxial[0] = 0.00000000000001

# Align cell
# cell.set_rotation(x=4.99, y=-4.33, z=3.14)
info_mea = {'electrode_name': 'nn_emi', 'pos': sites, 'center': False}
nn = mea.return_mea(info=info_mea)
pos_mea = nn.positions

# shift mea
# z_shift = [0, 0, np.max(pos_mea[:, 2]) - np.max(sites[:, 2])]
# shifted_pos = pos_mea - z_shift
# order = order_recording_sites(pos_mea - z_shift, sites)
pos = sites

stim_area = np.pi * cell.diam[cell.get_idx('dend')[0]] * cell_parameters['max_nsegs_length'] #um^2
I_max = 50 # (from EMI)

# Define synapse parameters
synapse_parameters = {
    'idx' : cell.get_closest_idx(x=0., y=0., z=350.),
    'e' : 0.,                   # reversal potential
    'syntype' : 'ExpSyn',       # synapse type
    'tau' : 2.,                 # synaptic time constant
    'weight' : 0.046,           # synaptic weight
    'record_current' : True,    # record synapse current
}

# Create synapse and set time of synaptic input
synapse = LFPy.Synapse(cell, **synapse_parameters)
synapse.set_spike_times(np.array([0.01]))
# synapse.set_spike_times(np.array([0.00]))

N = np.empty((pos.shape[0], 3))
n = 50
for i in range(N.shape[0]):
    N[i, ] = [1, 0, 0]  # normal vec. of contacts

# Add square electrodes (instead of circles)
if n > 1:
    electrode_parameters = {
        'sigma': 0.3,  # extracellular conductivity
        'x': pos[:, 0],  # x,y,z-coordinates of contact points
        'y': pos[:, 1],
        'z': pos[:, 2],
        'n': n,
        'r': 7.5,
        'N': N,
        'contact_shape': 'circle',
        'method': 'pointsource'
    }

ref_electrode_param = {
        'sigma': 0.3,  # extracellular conductivity
        'x': 0,  # x,y,z-coordinates of contact points
        'y': 0,
        'z': 500,
        'n': 1,
}


# Run simulation, electrode object argument in cell.simulate
print("running simulation...")
cell.simulate(rec_imem=True, rec_vmem=True, rec_ipas=True, rec_icap=True)

# Create electrode objects
electrode = LFPy.RecExtElectrode(cell,**electrode_parameters)
ref_electrode = LFPy.RecExtElectrode(cell,**ref_electrode_param)


# Calculate LFPs
electrode.calc_lfp()
ref_electrode.calc_lfp()
v_ext = electrode.LFP * 1000 #- ref_electrode.LFP * 1000

processing_time = time.time() - t_start
print('Processing time: ', processing_time)


mea.plot_mea_recording(v_ext, nn, time=end_T)

v_noprobe = np.load(join(no_mesh, 'v_probe.npy'))*1000
v_wprobe = np.load(join(w_mesh, 'v_probe.npy'))*1000


v_p_noprobe = np.squeeze(np.array([v_noprobe, v_ext]))
v_p_wprobe = np.squeeze(np.array([v_wprobe, v_ext, v_ext*2]))

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig1 = plt.figure(figsize=figsize)
ax1 = fig1.add_subplot(1,1,1)
ax1 = mea.plot_mea_recording(v_p_noprobe, nn, ax=ax1, time=end_T, lw=2, colors=[colors[0], colors[3]],
                             vscale=40, scalebar=True)
ax1.legend(labels=['EMI no probe', 'Cable Equation'], fontsize=fs_legend, loc='upper right')
fig1.tight_layout()

fig2 = plt.figure(figsize=figsize)
ax2 = fig2.add_subplot(1,1,1)
ax2 = mea.plot_mea_recording(v_p_wprobe, nn, ax=ax2, time=end_T, lw=2, colors=[colors[1], colors[3], colors[2]],
                             vscale=40, scalebar=True)
ax2.legend(labels=['EMI with probe', 'Cable Equation', 'Cable Equation + MoI'], fontsize=fs_legend, loc='upper right')
fig2.tight_layout()

print('NEURON min at: ', np.unravel_index(v_ext.argmin(), v_ext.shape))
print('EMI min at: ', np.unravel_index(v_noprobe.argmin(), v_noprobe.shape))

print('peak NEURON: ', np.min(v_ext))
print('peak EMI noprobe: ', np.min(v_noprobe))
print('difference noprobe: ', np.min(v_noprobe) - np.min(v_ext))
print('peak EMI wprobe: ', np.min(v_wprobe))
print('difference wprobe: ', np.min(v_wprobe) - np.min(v_ext))


# if save_fig:
#     fig1.savefig(join('../figures', 'bas_emi_noprobe_EAP.pdf'))
#     fig2.savefig(join('../figures', 'bas_emi_wprobe_EAP.pdf'))


