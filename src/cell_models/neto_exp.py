import numpy as np
import matplotlib.pylab as plt
import LFPy
import neuron
import MEAutility as mea
from neuroplot import *

neuron.h.load_file("stdrun.hoc")
neuron.h.load_file("import3d.hoc")

def get_templatename(f):
    '''
    Assess from hoc file the templatename being specified within

    Arguments
    ---------
    f : file, mode 'r'

    Returns
    -------
    templatename : str

    '''
    templatename = None
    f = file("template.hoc", 'r')
    for line in f.readlines():
        if 'begintemplate' in line.split():
            templatename = line.split()[-1]
            print 'template {} found!'.format(templatename)
            continue
    return templatename


def return_cell(cell_folder, cell_name, end_T, dt, start_T,add_synapses=False):
    """
    Function to load Human Brain Project cell models
    :param cell_folder: where the cell model is stored
    :param cell_name: name of the cell
    :param end_T: simulation length [ms]
    :param dt: time resoultion [ms]
    :param start_T: simulation start time (recording starts at 0 ms)
    :return: LFPy cell object
    """
    cwd = os.getcwd()
    os.chdir(cell_folder)
    print "Simulating ", cell_name

    # import neuron.hoc
    # del neuron.h
    # neuron.h = neuron.hoc.HocObject()

    neuron.load_mechanisms('../mods')

    f = file("template.hoc", 'r')
    templatename = get_templatename(f)
    f.close()

    f = file("biophysics.hoc", 'r')
    biophysics = get_templatename(f)
    f.close()

    f = file("morphology.hoc", 'r')
    morphology = get_templatename(f)
    f.close()

    #get synapses template name
    f = file(join("synapses", "synapses.hoc"), 'r')
    synapses = get_templatename(f)
    f.close()

    print('Loading constants')
    neuron.h.load_file('constants.hoc')
    print('...done.')
    if not hasattr(neuron.h, morphology):
        print 'loading morpho...'
        neuron.h.load_file(1, "morphology.hoc")
        print 'done.'

    if not hasattr(neuron.h, biophysics):
        neuron.h.load_file(1, "biophysics.hoc")

    if not hasattr(neuron.h, synapses):
        # load synapses
        neuron.h.load_file(1, os.path.join('synapses', 'synapses.hoc'))

    if not hasattr(neuron.h, templatename):
        print 'Loading template...'
        neuron.h.load_file(1, "template.hoc")
        print 'done.'

    morphologyfile = os.listdir('morphology')[0]#glob('morphology\\*')[0]

    # ipdb.set_trace()
    # Instantiate the cell(s) using LFPy
    print('Initialize cell...')
    cell = LFPy.TemplateCell(morphology=join('morphology', morphologyfile),
                     templatefile=os.path.join('template.hoc'),
                     templatename=templatename,
                     templateargs=1 if add_synapses else 0,
                     tstop=end_T,
                     tstart=start_T,
                     dt=dt,
                     v_init=-70,
                     pt3d=True,
                     delete_sections=True,
                     verbose=True)
    print('...done.')

    os.chdir(cwd)
    return cell

def set_input(weight, dt, T, cell, delay, stim_length):
    """
    Set current input synapse in soma
    :param weight: strength of input current [nA]
    :param dt: time step of simulation [ms]
    :param T: Total simulation time [ms]
    :param cell: cell object from LFPy
    :param delay: when to start the input [ms]
    :param stim_length: duration of injected current [ms]
    :return: NEURON vector of input current, cell object, and synapse
    """

    tot_ntsteps = int(round(T / dt + 1))

    I = np.ones(tot_ntsteps) * weight
    #I[stim_idxs] = weight
    noiseVec = neuron.h.Vector(I)
    syn = None
    for sec in cell.allseclist:
        if 'soma' in sec.name():
            # syn = neuron.h.ISyn(0.5, sec=sec)
            syn = neuron.h.IClamp(0.5, sec=sec)
    syn.dur = stim_length
    syn.delay = delay  # cell.tstartms
    noiseVec.play(syn._ref_amp, dt)

    return noiseVec, cell, syn


def find_spike_idxs(v, thresh=-30):
    """
    :param v: membrane potential
    :return: Number of zero-crossings in the positive direction, i.e., number of spikes
    """
    spikes = [idx for idx in range(len(v) - 1) if v[idx] < thresh < v[idx + 1]]
    return spikes


end_T = 500.
dt = 2**-5
start_T = 0.
figsize = (9, 14)
save_fig = True
plot_fig = False

cell_model = 'L5_TTPC1_cADpyr232_1'
cell = return_cell(cell_model, cell_model, end_T, dt, start_T)

mea_pos, dim, pitch = mea.return_site_positions('Neuronexus-32')

# rotate position
M = rotation_matrix2([0, 1., 0.], np.deg2rad(-48.2))
pos = np.dot(M, mea_pos.T).T

stim_length = 500
weight = 1
delay = 0
# weight = -1.25


noiseVec, cell, syn = set_input(weight, dt, end_T, cell, delay, stim_length)
# Define synapse parameters
# synapse_parameters = {
#     'idx' : cell.get_closest_idx(x=0., y=0., z=0.),
#     'e' : 0.,                   # reversal potential
#     'syntype' : 'ExpSyn',       # synapse type
#     'tau' : 2.,                 # synaptic time constant
#     'weight' : 100,              # synaptic weight
#     'record_current' : True,    # record synapse current
# }
#
# # Create synapse and set time of synaptic input
# synapse = LFPy.Synapse(cell, **synapse_parameters)
# synapse.set_spike_times(np.array([0.01]))

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

# from Neto
soma_half_size = 7.5
error = 0
min_dist = 31-soma_half_size-error
# closest-electrode 29 -> 0
closest = pos[12]

normal = np.abs(np.cross(pos[1]-pos[0], pos[-1]-pos[0]))
normal /= np.linalg.norm(normal)
new_pos = pos[12] + min_dist*normal

cell.set_pos(new_pos[0], new_pos[1], new_pos[2])


# Run simulation, electrode object argument in cell.simulate
print("running simulation...")
cell.simulate(rec_imem=True, rec_vmem=True, rec_ipas=True, rec_icap=True)
spike_idx = find_spike_idxs(cell.somav)

# Create electrode objects
electrode = LFPy.RecExtElectrode(cell,**electrode_parameters)

# Calculate LFPs
electrode.calc_lfp()
v_ext = electrode.LFP * 1000

# cut spikes
cut_out = [2. / dt, 3. / dt]
tspike = 7.
v_spikes = []
v_ext_spikes = []
i_spikes = []
plt.show()

for idx, spike_id in enumerate(spike_idx):
    spike_id = int(spike_id)
    v_spikes.append(cell.somav[spike_id - int(cut_out[0]):spike_id + int(cut_out[1])])
    i_spikes.append(cell.imem[:, spike_id - int(cut_out[0]):spike_id + int(cut_out[1])])
    v_ext_spikes.append(v_ext[:, spike_id - int(cut_out[0]):spike_id + int(cut_out[1])])

v_spikes = np.array(v_spikes)
v_ext_spikes = np.array(v_ext_spikes)
i_spikes = np.array(i_spikes)

colors = plt.rcParams['axes.color_cycle']
fig1 = plt.figure(figsize=figsize)
ax1 =  fig1.add_subplot(111)
mean_waveform = np.mean(v_ext_spikes, axis=0)
plot_mea_recording(mean_waveform, pos, pitch, time=tspike, ax=ax1, lw=2, colors=colors[1], vscale=300)

fig2 = plt.figure(figsize=figsize)
ax2 = fig2.add_subplot(111)
plot_mea_recording(v_ext, pos, pitch, time=end_T, vscale=300, ax=ax2)

fig1.tight_layout()
fig2.tight_layout()

#plot currents
# fig = plt.figure()
# ax1 = fig.add_subplot(1,3,1)
# [ax1.plot(cell.tvec, cell.imem[i]) for i in cell.get_idx('dend')]
# ax1.set_title('transmembrane total currents')
# plt.plot(np.transpose(v_ext))
# #plot currents
# ax2 = fig.add_subplot(1,3,2)
# [ax2.plot(cell.tvec, cell.ipas[i]) for i in cell.get_idx('dend')]
# ax2.set_title('passive')
#
# #plot currents
# ax3 = fig.add_subplot(1,3,3)
# [ax3.plot(cell.tvec, cell.icap[i]) for i in cell.get_idx('dend')]
# ax3.set_title('cap')
rec_peak = 421.

print np.min(v_ext), np.min(mean_waveform)
print np.max(np.abs(mean_waveform))/rec_peak * 100, ' %'

if plot_fig:
    plt.ion()
    plt.show()

if save_fig:
    fig1.savefig(join('../figures', 'neto_avg.pdf'))
    fig2.savefig(join('../figures', 'neto_eaps.pdf'))

print("done")

