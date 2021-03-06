import neuronmi
import numpy as np
import MEAutility as mu
import matplotlib.pylab as plt

microwire = True
neuronexus = True
neuropixels = True

plot_figures = True
save_figures = True

# mesh_resolution = {'neuron': 2, 'probe': 10, 'ext': 15}
mesh_resolution = 5

# define problem params

problem_parameters = {
    'neurons':  # List or dict. If list, length must be the same of number of neurons in mesh. If dict, the same
    # params are used for all neurons in the mesh
        {
            'cond_int': 7,  # float: Intracellular conductivity in mS/cm^2
            'Cm': 1.0,  # float: Membrane capacitance in uF/um^2
            'models': {},  # dict: Models for neuron domains. Default: {'dendrite': 'pas', 'soma': 'hh',
            #                                             'axon': 'hh'}
            'model_args': {},  # dict of tuples: Overwrite model default arguments.
            # E.g. ('dendrite', 'g_L'), ('soma', g_Na)'
            'stimulation': {'type': 'syn',  # str: Stimulation type ('syn', 'step', 'pulse')
                            'start_time': 0.01,  # float: Start of stimulation in ms
                            'stop_time': 1.0,  # float: Stop of stimulation in ms (it type is 'pulse')
                            'syn_weight': 10.0,  # float: Synaptic weight in in mS/cm^2 (if 'type' is 'syn')
                            'stim_current': 10,  # float: Stimulation current in nA (if 'type' is 'step'/'pulse')
                            'position': [0, 0, 350],  # array: 3D position of stimulation point in um.
                            'length': 20,  # float: length of stimulated portion in um.
                            'radius': 5,  # float: radius of stimulated area
                            }
        },
    'ext': {
        'cond_ext': 3,  # float: Extracellular conductivity: mS/cm^2
        'insulated_bcs': []  # list: Insulated BC for bounding box. It can be: 'max_x', 'min_x', etc.
    },
    'probe': {
        'stimulated_sites': None,  # List or tuple: Stimulated electrodes (e.g. [contact_3, contact_22]
        'type': None,  # str: Stimulation type ('step', 'pulse')
        'start_time': 0.1,  # float: Start of stimulation in ms
        'stop_time': 1.0,  # float: Stop of stimulation in ms (it type is 'pulse')
        'current': 0  # float: Stimulation current in mA. If list, each value correspond to a stimulated
        #        site
    },
    'solver': {
        'dt_fem': 0.01,  # float: dt for fem solver in ms
        'dt_ode': 0.01,  # float: dt for ode solver in ms
        'sim_duration': 5,  # float: duration od the simulation in ms
    }
}

neuron = neuronmi.mesh.shapes.TaperedNeuron({'dend_len': 420, 'axon_len': 210, 'axon_rad': 1, 'axonh_rad': 2})
box_size = {'xlim': [-110, 110], 'ylim': [-110, 110], 'zlim': [-260, 480]}

if microwire:
    # Microwire
    microwire_folder = 'probe_effect/microwire/'
    microwire_probe = neuronmi.mesh.shapes.MicrowireProbe({'tip_x': 30})
    mesh_with = neuronmi.generate_mesh(neurons=neuron, probe=microwire_probe,
                                       mesh_resolution=mesh_resolution, box_size=box_size,
                                       save_mesh_folder=microwire_folder)

    mesh_without = neuronmi.generate_mesh(neurons=neuron, probe=None, mesh_resolution=mesh_resolution,
                                          box_size=box_size,
                                          save_mesh_folder=microwire_folder)

    centers_microwire = microwire_probe.get_electrode_centers(unit='cm')

    u_with, _, _ = neuronmi.simulate_emi(mesh_with, u_probe_locations=centers_microwire,
                                         problem_params=problem_parameters)
    u_without, _, _ = neuronmi.simulate_emi(mesh_without, u_probe_locations=centers_microwire,
                                            problem_params=problem_parameters)

    np.save(microwire_folder + 'u_with.npy', u_with)
    np.save(microwire_folder + 'u_without.npy', u_without)
    np.save(microwire_folder + 'centers.npy', centers_microwire)

    if plot_figures:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(u_without, color='C0', lw=1.5, label='no probe')
        ax.plot(u_with, color='C1', lw=1.5, label='with probe')
        ax.legend(loc=3)
        if save_figures:
            fig.savefig(microwire_folder + 'microwire.pdf')

if neuronexus:
    # Neuronexus
    neuronexus_folder = 'probe_effect/neuronexus/'
    neuronexus_probe = neuronmi.mesh.shapes.NeuronexusProbe({'tip_x': 30})
    mesh_with = neuronmi.generate_mesh(neurons=neuron, probe=neuronexus_probe, mesh_resolution=mesh_resolution,
                                       box_size=box_size,
                                       save_mesh_folder=neuronexus_folder)

    mesh_without = neuronmi.generate_mesh(neurons=neuron, probe=None, mesh_resolution=mesh_resolution,
                                          box_size=box_size,
                                          save_mesh_folder=neuronexus_folder)

    centers_neuronexus = neuronexus_probe.get_electrode_centers(unit='cm')

    u_with, _, _ = neuronmi.simulate_emi(mesh_with, u_probe_locations=centers_neuronexus,
                                         problem_params=problem_parameters)
    u_without, _, _ = neuronmi.simulate_emi(mesh_without, u_probe_locations=centers_neuronexus,
                                            problem_params=problem_parameters)

    np.save(neuronexus_folder + 'u_with.npy', u_with)
    np.save(neuronexus_folder + 'u_without.npy', u_without)
    np.save(neuronexus_folder + 'centers.npy', centers_neuronexus)

    if plot_figures:
        probe = mu.return_mea(info={'pos': centers_neuronexus})
        vscale = np.max(np.abs(u_with))
        ax = mu.plot_mea_recording(u_without, probe, vscale=vscale, colors='C0', lw=1.5)
        ax = mu.plot_mea_recording(u_with, probe, vscale=vscale, colors='C1', ax=ax, lw=1.5)
        if save_figures:
            fig = ax.get_figure()
            fig.savefig(neuronexus_folder + 'neuronexus.pdf')

if neuropixels:
    # Neuropixels
    neuropixels_folder = 'probe_effect/neuropixels/'
    neuropixels_probe = neuronmi.mesh.shapes.Neuropixels24Probe({'tip_x': 30})
    mesh_with = neuronmi.generate_mesh(neurons=neuron, probe=neuropixels_probe, mesh_resolution=mesh_resolution,
                                       box_size=box_size,
                                       save_mesh_folder=neuropixels_folder)

    mesh_without = neuronmi.generate_mesh(neurons=neuron, probe=None, mesh_resolution=mesh_resolution,
                                          box_size=box_size,
                                          save_mesh_folder=neuropixels_folder)

    centers_neuropixels = neuropixels_probe.get_electrode_centers(unit='cm')

    u_with, _, _ = neuronmi.simulate_emi(mesh_with, u_probe_locations=centers_neuropixels,
                                         problem_params=problem_parameters)
    u_without, _, _ = neuronmi.simulate_emi(mesh_without, u_probe_locations=centers_neuropixels,
                                            problem_params=problem_parameters)

    np.save(neuropixels_folder + 'u_with.npy', u_with)
    np.save(neuropixels_folder + 'u_without.npy', u_without)
    np.save(neuropixels_folder + 'centers.npy', centers_neuropixels)

    if plot_figures:
        probe = mu.return_mea(info={'pos': centers_neuropixels})
        vscale = np.max(np.abs(u_with))
        ax = mu.plot_mea_recording(u_without, probe, vscale=vscale, colors='C0', lw=1.5)
        ax = mu.plot_mea_recording(u_with, probe, vscale=vscale, colors='C1', ax=ax, lw=1.5)
        if save_figures:
            fig = ax.get_figure()
            fig.savefig(neuropixels_folder + 'neuropixels.pdf')
