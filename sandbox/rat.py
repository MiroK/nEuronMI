from neuronmi.mesh.mesh_utils import load_h5_mesh
import numpy as np
import neuronmi
import os


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
        'dt_ode': 0.001,  # float: dt for ode solver in ms
        'sim_duration': 5,  # float: duration od the simulation in ms
    }
}

pde_formulation = 'pm'

mesh_without = './rat_6mesh/RatS1-6-39.CNG.c2_tagged.h5'
scale_factor = 1E-4
mesh, _, _ = load_h5_mesh(mesh_without, scale_factor)

p_min, p_max = mesh.coordinates().min(axis=0), mesh.coordinates().max(axis=0)
dx = p_max - p_min
r = min(dx[0], dx[1])
probes = np.c_[0.4*r*np.ones(30),
               0.4*r*np.ones(30),
               np.linspace(p_min[2], p_max[2], 32)[1:-1]]


folder = os.path.dirname(mesh_without)
u_without, _ = neuronmi.simulate_emi(folder, u_probe_locations=probes,
                                     problem_params=problem_parameters,
                                     pde_formulation=pde_formulation,
                                     save_folder=folder, save_format='xdmf')

np.save(folder + 'u_without.npy', u_without)
np.save(folder + 'centers.npy', probes)
