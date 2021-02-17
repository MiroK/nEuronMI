import numpy as np
import MEAutility as mu
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path
from matplotlib import cm
from matplotlib.colors import Normalize

import argparse


def load_data(directory, probe_name, extrema=False):
    """Return probe array and positions. """
    probe_directory = Path(directory) / probe_name
    array = np.loadtxt(str(probe_directory / "{}.txt".format(probe_name)), delimiter=",")
    positions = np.loadtxt(str(probe_directory / "{}_coordinates.txt".format(probe_name)))

    if extrema:
        for location, amin, amax in zip(positions, np.min(array, axis=0), np.max(array, axis=0)):
            print("pos: {}, min: {}, max: {}".format(location, amin, amax))
    return array, positions


def plot_mea(array, positions, show=False):
    array = array[:, 2:-2] + 75     # TODO: Experiment with the slicing. How much can I include?
    probe = mu.return_mea(info={'pos': positions[5:25]})
    ax = mu.plot_mea_recording(array.T, probe)
    if show:
        plt.show()
    return ax.get_figure()


def plot_centreline(data_array, coordinates, show):
    """Plot the potential across the neuron.

    Plot potential samples from diferent location of the neuronal centreline. The lines are
    colored based on where they are. Blac lines for the soma. Green lines for the dendrite and red
    lines for the axon.
    """
    data_array = data_array[:, 2:-2] + 75     # TODO: Experiment with the slicing. How much can I include?
    data_array *= 1000      # convert to micro volts
    fig, ax = plt.subplots()

    soma_radius = 10/10000      # from default ball and stick parameters

    # TODO: All these are paramter specific values from default ball and stick parameters
    norm1 = Normalize(-85/10000, -10/10000)
    norm2 = Normalize(-10/1000, 10/1000)        # darker scale for the soma
    norm3 = Normalize(10/10000, 335/10000)

    for array, coordinate in zip(data_array.T, coordinates):
        if coordinate[-1] < -soma_radius:       # axon
            # Red colors for the axon. Darker colors away from the soma
            colormap = cm.get_cmap("autumn")
            color = colormap(norm1(coordinate[-1]))
        elif coordinate[-1] < soma_radius:      # soma!
            colormap = cm.get_cmap("twilight")
            color = colormap(norm2(coordinate[-1]))
        else:                                   # dendrite
            # Green lines for the dendrite. Reverse colormap so darker colors far from soma
            colormap = cm.get_cmap("summer_r")
            color = colormap(norm3(coordinate[-1]))
        ax.plot(array, color=color)

    ax.set_xlabel("time")       # What unit?
    ax.set_ylabel("Potential [$\mu V$]")
    if show:
        plt.show()


def plot_extracellular(data_array, positions, d, show):
    """TODO: Experimental and very neuron geometry parameter dependent."""
    fig, ax = plt.subplots()
    data_array = data_array[:, 1:]

    rr = (10 + 2.5)/2     # half of some R + dendrtic r
    indices = ((-d/2 + rr)/10000 < positions[:, 1]) & (positions[:, 1] < (d/2 + rr) / 10000)

    minimum = np.min(positions[indices, 1])
    maximum = np.max(positions[indices, 1])
    norm = Normalize(minimum, maximum)
    colormap = cm.get_cmap("hot")

    # assert False, "It looks good, but I have to make sure the colors are right. Darker away from the "
    for array, pos in zip((data_array.T)[indices], norm(positions[indices, 1])):
        ax.plot(array, color=colormap(pos))
    if show:
        plt.show()


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-pp",
        "--probe-path",
        help="Path to probe",
        required=True,
    )

    parser.add_argument(
        "-n",
        "--name",
        help="Name of probe",
        required=True
    )

    parser.add_argument(
        "--show",
        help="Display plot",
        action="store_true",
        required=False
    )

    parser.add_argument(
        "--plot-mea",
        help="Plot the probes using meautility",
        action="store_true",
        required=False
    )

    parser.add_argument(
        "--extracellular-distance",
        help="Plot the potential across the extracellular space between the neurons. NB! Experimental!",
        required=False,
        default=None
    )

    parser.add_argument(
        "--output",
        help="Name under which to store the output figure. The '.png' extension is assumed.",
        required=False,
        default=None
    )

    parser.add_argument(
        "--print-extrema",
        help="Print the max and min values for each probe location.",
        required=False,
        action="store_true",
    )

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    array, positions = load_data(args.probe_path, args.name, args.print_extrema)

    # Do nor plot centreline if probes are extracellular
    if args.extracellular_distance is None:
        fig_centreline = plot_centreline(array, positions, args.show)
        if args.output is not None:
            fig_centreline.savefig("{}_centreline.png".format(args.output))
    else:
        fig_extrcellular = plot_extracellular(
            array,
            positions,
            int(args.extracellular_distance),
            args.show
        )
        if args.output is not None:
            fig_extrcellular.savefig("{}_extracellular.png".format(args.output))

    if args.plot_mea:
        fig_mea = plot_mea(array, positions, args.show)
        if args.output is not None:
            fig_mea.savefig("{}_mea.png".format(args.output))



if __name__ == "__main__":
    main()
