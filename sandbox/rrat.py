from neuronmi.mesh.mesh_utils import load_h5_mesh, find_soma, ReducedEMIEntityMap
import numpy as np
import neuronmi
import os

from rat import problem_parameters

# FIXME: update syn_weigth

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

    u_probes = [ue_probes] + ui_probes


    # Now with probes set
    folder = './rat_6reduced'
    folder = os.path.dirname(mesh_without)

    u_without = neuronmi.reduced_simulate_emi(folder, u_probe_locations=u_probes,
                                              problem_params=problem_parameters,
                                              save_folder=folder, save_format='xdmf')
    
    np.save(folder + 'u_without.npy', u_without)
    np.save(folder + 'centers.npy', u_probes)
