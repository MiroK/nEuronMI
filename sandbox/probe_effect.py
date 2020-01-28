import neuronmi
import numpy as np

microwire = False
neuronexus = True
neuropixels = False

# define problem params

problem_parameters = {
    'neurons':  # List or dict. If list, length must be the same of number of neurons in mesh. If dict, the same
                # params are used for all neurons in the mesh
                {
                 'cond_int': 7,           # float: Intracellular conductivity in mS/cm^2
                 'Cm': 1.0,               # float: Membrane capacitance in uF/um^2
                 'models': {},            # dict: Models for neuron domains. Default: {'dendrite': 'pas', 'soma': 'hh',
                                          #                                             'axon': 'hh'}
                 'model_args': {},        # dict of tuples: Overwrite model default arguments.
                                          # E.g. ('dendrite', 'g_L'), ('soma', g_Na)'
                 'stimulation': {'type': 'syn',           # str: Stimulation type ('syn', 'step', 'pulse')
                                 'start_time': 0.01,       # float: Start of stimulation in ms
                                 'stop_time': 1.0,        # float: Stop of stimulation in ms (it type is 'pulse')
                                 'strength': 10.0,        # float: Stimulation strength in mS/cm^2
                                 'position': 350,         # float or array: position of stimulation in um. DEPRECATED
                                 'length': 20             # float: length od stimulated portion in um. DEPRECATED
                 }
    },
    'ext':     {
                 'cond_ext': 3,           # float: Extracellular conductivity: mS/cm^2
                 'insulated_bcs': []      # list: Insulated BC for bounding box. It can be: 'max_x', 'min_x', etc.
    },
    'probe':   {
                 'stimulated_sites': None,  # List or tuple: Stimulated electrodes (e.g. [contact_3, contact_22]
                 'type': None,            # str: Stimulation type ('step', 'pulse')
                 'start_time': 0.1,       # float: Start of stimulation in ms
                 'stop_time': 1.0,        # float: Stop of stimulation in ms (it type is 'pulse')
                 'current': 0             # float: Stimulation current in mA. If list, each value correspond to a stimulated
                                          #        site
    },
    'solver':  {
                 'dt_fem': 0.01,          # float: dt for fem solver in ms
                 'dt_ode': 0.01,          # float: dt for ode solver in ms
                 'sim_duration': 5,      # float: duration od the simulation in ms
    }
}

neuron = neuronmi.mesh.shapes.TaperedNeuron({'dend_len': 420, 'axon_len': 210, 'axon_rad': 1, 'axonh_rad': 2})
box_size = {'xlim': [-110, 110], 'ylim': [-110, 110], 'zlim': [-260, 480]}

if microwire:
    # Microwire
    microwire_folder = 'probe_effect/microwire/'
    microwire_probe = neuronmi.mesh.shapes.MicrowireProbe({'tip_x': 40})
    mesh_with = neuronmi.generate_mesh(neurons=neuron, probe=microwire_probe, mesh_resolution=5, box_size=box_size,
                                       save_mesh_folder=microwire_folder)

    mesh_without = neuronmi.generate_mesh(neurons=neuron, probe=None, mesh_resolution=5, box_size=box_size,
                                          save_mesh_folder=microwire_folder)

    centers_microwire = microwire_probe.get_electrode_centers(unit='cm')

    u_with, _ = neuronmi.simulate_emi(mesh_with, u_probe_locations=centers_microwire, problem_params=problem_parameters)
    u_without, _ = neuronmi.simulate_emi(mesh_without, u_probe_locations=centers_microwire, problem_params=problem_parameters)

    np.save(microwire_folder + 'u_with.npy', u_with)
    np.save(microwire_folder + 'u_without.npy', u_without)
    np.save(microwire_folder + 'centers.npy', centers_microwire)

if neuronexus:
    # Neuronexus
    neuronexus_folder = 'probe_effect/neuronexus/'
    neuronexus_probe = neuronmi.mesh.shapes.NeuronexusProbe({'tip_x': 40})
    mesh_with = neuronmi.generate_mesh(neurons=neuron, probe=neuronexus_probe, mesh_resolution=5, box_size=box_size,
                                       save_mesh_folder=neuronexus_folder)

    mesh_without = neuronmi.generate_mesh(neurons=neuron, probe=None, mesh_resolution=5, box_size=box_size,
                                          save_mesh_folder=neuronexus_folder)

    centers_neuronexus = neuronexus_probe.get_electrode_centers(unit='cm')

    u_with, _ = neuronmi.simulate_emi(mesh_with, u_probe_locations=centers_neuronexus, problem_params=problem_parameters)
    u_without, _ = neuronmi.simulate_emi(mesh_without, u_probe_locations=centers_neuronexus, problem_params=problem_parameters)

    np.save(neuronexus_folder + 'u_with.npy', u_with)
    np.save(neuronexus_folder + 'u_without.npy', u_without)
    np.save(neuronexus_folder + 'centers.npy', centers_neuronexus)

if neuropixels:
    # Neuropixels
    neuropixels_folder = 'probe_effect/neuropixels/'
    neuropixels_probe = neuronmi.mesh.shapes.Neuropixels24Probe({'tip_x': 40})
    mesh_with = neuronmi.generate_mesh(neurons=neuron, probe=neuropixels_probe, mesh_resolution=5, box_size=box_size,
                                       save_mesh_folder=neuropixels_folder)

    mesh_without = neuronmi.generate_mesh(neurons=neuron, probe=None, mesh_resolution=5, box_size=box_size,
                                          save_mesh_folder=neuropixels_folder)

    centers_neuropixels = neuropixels_probe.get_electrode_centers(unit='cm')

    u_with, _ = neuronmi.simulate_emi(mesh_with, u_probe_locations=centers_neuropixels, problem_params=problem_parameters)
    u_without, _ = neuronmi.simulate_emi(mesh_without, u_probe_locations=centers_neuropixels, problem_params=problem_parameters)

    np.save(neuropixels_folder + 'u_with.npy', u_with)
    np.save(neuropixels_folder + 'u_without.npy', u_without)
    np.save(neuropixels_folder + 'centers.npy', centers_neuropixels)
