import numpy as np
import dolfin as df

import neuronmi
import sys
import shutil

from pathlib import Path


def coordinates(neuron_params, num_centreline_points, num_soma_points, perpendicular_distance):
    """Create coordinates for probes."""
    soma_rad = neuron_params["soma_rad"]
    neuronal_centreline = np.zeros((num_centreline_points, 3))

    neuronal_centreline[:, 0] = neuron_params["soma_x"]
    neuronal_centreline[:, 1] = neuron_params["soma_y"]

    neuron_start = neuron_params["soma_z"] - soma_rad - neuron_params["axon_len"]
    neuron_stop = neuron_params["soma_z"] + soma_rad + neuron_params["dend_len"]
    neuronal_centreline[:, 2] =  np.linspace(neuron_start, neuron_stop, num_centreline_points)

    soma_cross_centreline = np.zeros((num_soma_points, 3))
    soma_cross_centreline[:, 0] = neuron_params["soma_x"]
    soma_cross_centreline[:, 2] = neuron_params["soma_z"]
    soma_cross_centreline[:, 1] = np.linspace(
        neuron_params["soma_y"] - perpendicular_distance,
        neuron_params["soma_y"] + perpendicular_distance,
        num_soma_points
    )

    neuronal_centreline /= 10000
    soma_cross_centreline /= 10000
    return neuronal_centreline, soma_cross_centreline


def run(parallel_distance, transverse_distance, box_size, mesh_resolution, neuron_parameters):
    # We are using the same basic parameters for both neurons
    p1 = neuron_parameters.copy()
    p2 = neuron_parameters.copy()

    p1["soma_y"] = -transverse_distance / 2
    p2["soma_y"] = transverse_distance / 2

    p1["soma_z"] = -parallel_distance / 2
    p2["soma_z"] = parallel_distance / 2

    mesh_folder = neuronmi.generate_mesh(
        neurons=['bas', 'bas'],
        neuron_params=[p1, p2],
        probe=None,
        mesh_resolution=mesh_resolution,
        box_size=box_size,
    )
    mesh_folder = Path(mesh_folder)

    neuron_params_0 = neuronmi.get_default_emi_params()['neurons'].copy()
    neuron_params_1 = neuronmi.get_default_emi_params()['neurons'].copy()

    neuron_params_0['stimulation']['type'] = "syn"      # defualt = 10
    neuron_params_1['stimulation']['type'] = "syn"

    neuron_params_0['stimulation']['syn_weight'] = 0.0       # defualt = 10
    neuron_params_1['stimulation']['syn_weight'] = 20.0

    params = neuronmi.get_default_emi_params()
    params['neurons'] = [neuron_params_0, neuron_params_1]
    params["solver"]["sim_duration"] = 1

    i_probes = [
        (p2["soma_x"], p2["soma_y"] + 10, p2["soma_z"]),
        (p2["soma_x"], p2["soma_y"] - 10, p2["soma_z"])
    ]

    u_probes = [
        (p2["soma_x"], p2["soma_y"], p2["soma_z"]),
        (p1["soma_x"], p1["soma_y"] - p1["soma_rad"], p1["soma_z"])
    ]

    u_records, i_records, v_records = neuronmi.simulate_emi(mesh_folder, params, i_probe_locations=i_probes, u_probe_locations=u_probes)
    print u_records, i_records
    np.savetxt(str(mesh_folder / "uprobes.txt"), u_records, delimiter=",")
    np.savetxt(str(mesh_folder / "iprobes.txt"), i_records, delimiter=",")
    print "Success!"


def main():
    neuron_parameters = neuronmi.get_neuron_params('bas')       # bas == ball and stick

    parallel_distance = -50
    try:
        d = int(sys.argv[1])
    except IndexError:
        d = 5
        print "No command line arguments, using d = %d" % d
    perpendicular_distance = neuron_parameters["soma_rad"] + neuron_parameters["dend_rad"] + d
    box_size = 4
    resolution = 5

    run(parallel_distance, perpendicular_distance, box_size, resolution, neuron_parameters)


if __name__ == "__main__":
    main()
