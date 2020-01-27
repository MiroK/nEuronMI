import numpy as np
import MEAutility as mu
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

from matplotlib.collections import PatchCollection


import argparse


def load_data(directory, probe_name, extrema=False):
    """Return probe array and positions. """
    probe_directory = Path(directory) / probe_name
    array = np.loadtxt(str(probe_directory / "{}.txt".format(probe_name)), delimiter=",")
    positions = np.loadtxt(str(probe_directory / "{}_coordinates.txt".format(probe_name)))

    array = array[:, 2:-2] + 75     # TODO: Experiment with the slicing. How much can I include?

    if extrema:
        for location, amin, amax in zip(positions, np.min(array, axis=0), np.max(array, axis=0)):
            print("pos: {}, min: {}, max: {}".format(location, amin, amax))
    return array, positions


def plot_mea(array, positions, show=False):
    probe = mu.return_mea(info={'pos': positions[5:25]})
    ax = mu.plot_mea_recording(array.T, probe)
    if show:
        plt.show()
    return ax.get_figure()


def plot_all(array, show):
    fig, ax = plt.subplots()
    ax.plot(array)
    if show:
        plt.show()
    return fig




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
        "--plot-all",
        help="Plot all the lines with automtic scaling and colors",
        action="store_true",
        required=False
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
    fig = plot_mea(array, positions, args.show)
    if args.output is not None:
        fig.savefig("{}.png".format(args.output))

    if args.plot_all:
        fig_all = plot_all(array, args.show)
        if args.output is not None:
            fig_all.savefig("{}_all.png".format(args.output))


if __name__ == "__main__":
    main()
