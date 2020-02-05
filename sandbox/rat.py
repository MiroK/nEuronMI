import neuronmi
import numpy as np

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
                            'position': [0, 0, 100],  # array: 3D position of stimulation point in um.
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

# Neuronexus
neuronexus_probe = neuronmi.mesh.shapes.NeuronexusProbe({'tip_x': 30})

centers_neuronexus = neuronexus_probe.get_electrode_centers(unit='cm')
print centers_neuronexus
mesh_without = './rat_6mesh'
folder = mesh_without
u_without, _ = neuronmi.simulate_emi(mesh_without, u_probe_locations=centers_neuronexus,
                                     problem_params=problem_parameters,
                                     pde_formulation='pm',
                                     save_folder=folder, save_format='xdmf')

#np.save(neuronexus_folder + 'u_with.npy', u_with)
#np.save(neuronexus_folder + 'u_without.npy', u_without)
#np.save(neuronexus_folder + 'centers.npy', centers_neuronexus)#
