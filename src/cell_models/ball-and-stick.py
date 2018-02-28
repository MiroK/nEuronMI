import numpy as np
import matplotlib.pylab as plt
import LFPy
import neuron
import yaml
import MEAutility as mea
from neuroplot import *
import time

plt.ion()
plt.show()

def order_recording_sites(sites1, sites2):
    pairs = []
    for i_s1, s1 in enumerate(sites1):
        distances = [np.linalg.norm(s1 - s2) for s2 in sites2]
        pairs.append([i_s1, np.argmin(distances)])

    return np.array(pairs)

t_start = time.time()

end_T = 5.

# Define cell parameters
cell_parameters = {
    'morphology' : 'ball_and_stick_waxon.hoc', # from Mainen & Sejnowski, J Comput Neurosci, 1996
    'cm' : 1.0,         # membrane capacitance
    'Ra' : 150.,        # axial resistance
    'v_init' : -75.,    # initial crossmembrane potential
    'passive' : True,   # turn on NEURONs passive mechanism for all sections
    # 'passive_parameters' : {'g_pas' : 1./30000, 'e_pas' : -75},
    'passive_parameters': {'g_pas': 0.06*1E-3, 'e_pas': -75},
    # 'nsegs_method': 'lambda100',  # spatial discretization method
    'nsegs_method' : 'fixed_length', # spatial discretization method
    'max_nsegs_length' : 1,
    'lambda_f' : 100.,           # frequency where length constants are computed
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

# Align cell
# cell.set_rotation(x=4.99, y=-4.33, z=3.14)

pos, dim, pitch = mea.return_site_positions('Neuronexus-32')
pos[:, 0] += 32.5
pos[:, 2] += 99.5

stim_area = np.pi * cell.diam[cell.get_idx('dend')[0]] * cell_parameters['max_nsegs_length'] #um^2
I_max = 50 # (from EMI)



# Define synapse parameters
synapse_parameters = {
    'idx' : cell.get_closest_idx(x=0., y=0., z=350.),
    'e' : 0.,                   # reversal potential
    'syntype' : 'ExpSyn',       # synapse type
    'tau' : 2.,                 # synaptic time constant
    'weight' : 0.043,            # synaptic weight
    'record_current' : True,    # record synapse current
}

# Create synapse and set time of synaptic input
synapse = LFPy.Synapse(cell, **synapse_parameters)
synapse.set_spike_times(np.array([0.01]))

N = np.empty((pos.shape[0], 3))
n = 50
for i in xrange(N.shape[0]):
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


# Run simulation, electrode object argument in cell.simulate
print("running simulation...")
cell.simulate(rec_imem=True, rec_vmem=True, rec_ipas=True, rec_icap=True)

# Create electrode objects
electrode = LFPy.RecExtElectrode(cell,**electrode_parameters)

# Calculate LFPs
electrode.calc_lfp()
v_ext = electrode.LFP * 1000

processing_time = time.time() - t_start
print 'Processing time: ', processing_time


plot_mea_recording(v_ext, pos, pitch, time=end_T)

# #plot currents
# fig = plt.figure()
# ax1 = fig.add_subplot(1,3,1)
# [ax1.plot(cell.tvec, cell.imem[i]) for i in cell.get_idx('dend')]
# ax1.set_title('transmembrane total currents')
#
# #plot currents
# ax2 = fig.add_subplot(1,3,2)
# [ax2.plot(cell.tvec, cell.ipas[i]) for i in cell.get_idx('dend')]
# ax2.set_title('passive')
#
# #plot currents
# ax3 = fig.add_subplot(1,3,3)
# [ax3.plot(cell.tvec, cell.icap[i]) for i in cell.get_idx('dend')]
# ax3.set_title('cap')


### Compare with EMI ###
no_mesh = '../results/mainen_fancy_40_0_-100_coarse_0_box_3_noprobe'

conv=1E-4
fs_legend = 20
save_fig = False
figsize = (9, 14)

with open(join(no_mesh, 'params.yaml'), 'r') as f:
    info = yaml.load(f)

T = info['problem']['Tstop']

times = np.load(join(no_mesh, 'times.npy'))
sites = np.load(join(no_mesh, 'sites.npy'))/conv

v_noprobe = np.load(join(no_mesh, 'v_probe.npy'))*1000

pairs = order_recording_sites(pos, sites)
v_ordered = v_noprobe[pairs[:, 1]]


v_p = np.squeeze(np.array([v_ordered, v_ext]))
# pitch = np.array([18., 25.])

colors = plt.rcParams['axes.color_cycle']
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(1,1,1)
ax = plot_mea_recording(v_p, pos, pitch, ax=ax, time=end_T, lw=2, colors=[colors[0], colors[3]], vscale=40)
ax.legend(labels=['EMI no probe', 'Cable Equation'], fontsize=fs_legend, loc='upper right')
fig.tight_layout()

print 'NEURON min at: ', np.unravel_index(v_ext.argmin(), v_ext.shape)
print 'EMI min at: ', np.unravel_index(v_ordered.argmin(), v_ordered.shape)

print 'peak NEURON: ', np.min(v_ext)
print 'peak EMI: ', np.min(v_ordered)
print 'difference: ', np.min(v_ordered) - np.min(v_ext)


if save_fig:
    fig.savefig(join('../figures', 'bas_emi_EAP.pdf'))
