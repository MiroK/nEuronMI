from neuronmi.mesh.mesh_utils import load_h5_mesh, find_soma, ReducedEMIEntityMap
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
                            'syn_weight': 50.0,  # float: Synaptic weight in in mS/cm^2 (if 'type' is 'syn')
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

# -------------------------------------------------------------------

if __name__ == '__main__':
    from xii import EmbeddedMesh
    
    # Where we will sample
    mesh_without = './rat_6reduced/RatS1-6-39.CNG_trimmed.c1.h5'
    scale_factor = 1E-4
    mesh, _, _, edge_marking_f = load_h5_mesh(mesh_without, scale_factor)

    mesh_json_path = './rat_6reduced/RatS1-6-39.CNG_trimmed.c1.json'
    with open(mesh_json_path, 'r') as json_fp:
        emi_map = ReducedEMIEntityMap(json_fp=json_fp)
    
    # We want to sample along centerlines
    num_neurons = emi_map.num_neurons

    neurons = [EmbeddedMesh(edge_marking_f, emi_map.curve_physical_tags('neuron_%d' % i).values())
               for i in range(num_neurons)]

    ui_probes = [scale_factor*neuron.coordinates() for neuron in neurons]

    ue_probes = None
    # For extracellular we will look around soma
    for i, neuron in enumerate(neurons):
        name = 'neuron_%d' % i
        center, rad = find_soma(neuron, emi_map.curve_types(name), emi_map.curve_radii(name))
        # Unit convert
        center, rad = center*scale_factor, rad*scale_factor
        arrow = np.array([1., 1, 0])
        arrow = arrow / np.linalg.norm(arrow)

        pts = [center + r*arrow for r in np.linspace(2*rad, 3*rad, 5)]
        if ue_probes is None:
            ue_probes = np.row_stack(pts)
        else:
            ue_probes = np.row_stack([ue_probes, pts])

    u_probes = np.row_stack([ue_probes] + ui_probes)


    # Now with probes set
    pde_formulation = 'pm'
    folder = os.path.dirname('./rat_6mesh/RatS1-6-39.CNG.c2_tagged.h5')
    u_without, _ = neuronmi.simulate_emi(folder, u_probe_locations=u_probes,
                                         problem_params=problem_parameters,
                                         pde_formulation=pde_formulation,
                                         save_folder=folder, save_format='xdmf')

    np.save(folder + 'u_without.npy', u_without)
    np.save(folder + 'centers.npy', u_probes)
