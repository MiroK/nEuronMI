import numpy as np
import dolfin as df

import neuronmi
import sys
import shutil

from pathlib import Path
from fenicstools import Probes

import argparse


def coordinates(neuron_params, num_centreline_points, num_soma_points):
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
    soma_cross_centreline[:, 1] = np.linspace(-soma_rad, soma_rad, num_soma_points)

    neuronal_centreline /= 10000
    soma_cross_centreline /= 10000
    return neuronal_centreline, soma_cross_centreline


class PointProbe:
    def __init__(self, name, coordinates, output_directory):
        """Wrapper class for probes. """
        self._coordinates = coordinates
        self._name = name
        self._output_directory = Path(output_directory)
        self._probes = None

    def _initialise(self, solution):
        """Initialise probes, and path infrastructure."""
        function_space = solution.function_space()
        self._probes = Probes(self._coordinates.flatten(), function_space)

        self._path = self._output_directory / self._name
        if self._path.exists():
            shutil.rmtree(str(self._path))
        self._path.mkdir(parents=True)
        self._filename = self._path / ("%s.txt" % self._name)
        self._filename_coordinates = self._path / ("%s_coordinates.txt" % self._name)

        np.savetxt(str(self._filename_coordinates), self._coordinates)

    def _compute(self, solution):
        """Sample the solution. """
        if self._probes is None:
            self._initialise(solution)

        self._probes(solution)
        result = self._probes.array()
        self._probes.clear()        # TODO: Why?
        return result

    def update(self, time, solution):
        """Store the solution."""
        data = self._compute(solution)

        with self._filename.open("a") as ofh:
            data_format = ", ".join(("%f",)*(len(data) + 1))
            line = data_format % tuple([time] + list(data))
            ofh.write(u"%s" % line)
            ofh.write(u"\n")


def run(parallel_distance, transverse_distance, box_size, mesh_resolution, neuron_parameters):
    # We are using the same basic parameters for both neurons
    p1 = neuron_parameters.copy()
    p2 = neuron_parameters.copy()

    p1["soma_y"] = transverse_distance / 2
    p2["soma_y"] = -transverse_distance / 2

    p1["soma_z"] = parallel_distance / 2
    p2["soma_z"] = -parallel_distance / 2

    mesh_folder = neuronmi.generate_mesh(
        neuron_type=['bas', 'bas'],
        neuron_params=[p1, p2],
        probe_type=None,
        mesh_resolution=mesh_resolution,
        box_size=box_size,
    )
    mesh_folder = Path(mesh_folder)

    neuron_params_0 = neuronmi.get_default_emi_params()['neurons']
    neuron_params_1 = neuronmi.get_default_emi_params()['neurons']

    neuron_params_0['stimulation']['strength'] = 20.0       # defualt = 10
    neuron_params_1['stimulation']['strength'] = 0

    params = neuronmi.get_default_emi_params()
    params['neurons'] = [neuron_params_0, neuron_params_1]

    # Set up probes
    p1_centreline_coordinates, p1_soma_coordinates = coordinates(p1, 30, 10)
    p2_centreline_coordinates, p2_soma_coordinates = coordinates(p2, 30, 10)

    p1_centreline_probe = PointProbe("p1_centreline", p1_centreline_coordinates, mesh_folder / "probes")
    p1_soma_probe = PointProbe("p1_soma", p1_soma_coordinates, mesh_folder / "probes")

    p2_centreline_probe = PointProbe("p2_centreline", p2_centreline_coordinates, mesh_folder / "probes")
    p2_soma_probe = PointProbe("p2_soma", p2_soma_coordinates, mesh_folder / "probes")

    probe_list = [p1_centreline_probe, p1_soma_probe, p2_centreline_probe, p2_soma_probe]

    xdmf_I = df.XDMFFile(str(mesh_folder / 'emi_sim' / 'I.xdmf'))
    # hdf5_I = df.HDF5File(df.MPI.comm_world, str(mesh_folder / 'emi_sim' / 'I.xdmf'), "w")

    xdmf_u = df.XDMFFile(str(mesh_folder / 'emi_sim' / 'u.xdmf'))
    # hdf5_u = df.HDF5File(df.MPI.comm_world, str(mesh_folder / 'emi_sim' / 'u.xdmf'), "w")

    for t, u, I in neuronmi.simulate_emi(mesh_folder, params):
        print "%.2f" % float(t)
        for probe in probe_list:
            probe.update(t, u)

        # hdf5_I.write()
        xdmf_I.write(I, float(t))

        # hdf5_u.write()
        xdmf_u.write(u, float(t))
    print "Success!"


def main():
    # neuron_parameters = neuronmi.mesh.shapes.BallStickNeuron._defaults
    neuron_parameters = neuronmi.get_neuron_params('bas')       # bas == ball and stick

    parallel_distance = -50
    try:
        d = int(sys.argv[1])
    except IndexError:
        print "No command line arguments, using d = 1"
        d = 1
    perpendicular_distance = neuron_parameters["soma_rad"] + neuron_parameters["dend_rad"] + d
    box_size = 4
    resolution = 5

    run(parallel_distance, perpendicular_distance, box_size, resolution, neuron_parameters)


if __name__ == "__main__":
    main()
