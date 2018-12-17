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

# load mesh
conv=1E-4
sites = np.loadtxt('fem_pos.txt')
t_start = time.time()

fs_legend = 20
save_fig = False
figsize = (9, 14)
end_T = 5.

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
info_mea = {'electrode_name': 'nn_emi', 'pos': sites, 'center': False}
nn = mea.return_mea(info=info_mea)
pos_mea = nn.positions
pos = sites

stim_area = np.pi * cell.diam[cell.get_idx('dend')[0]] * cell_parameters['max_nsegs_length'] #um^2
I_max = 50 # (from EMI)

# Define synapse parameters
synapse_parameters = {
    'idx' : cell.get_closest_idx(x=0., y=0., z=350.),
    'e' : 0.,                   # reversal potential
    'syntype' : 'ExpSyn',       # synapse type
    'tau' : 2.,                 # synaptic time constant
    'weight' : 0.0455,           # synaptic weight
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

cell.set_pos(-40, -30, 0)

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

print('NEURON min at: ', np.unravel_index(v_ext.argmin(), v_ext.shape))

np.savetxt('bas_imem_30.txt', cell.imem)
np.savetxt('bas_vext_30.txt', v_ext)
np.savetxt('elec_pos_30.txt', pos)
np.savetxt('seg_pos_30.txt', np.array([cell.xmid, cell.ymid, cell.zmid]))
