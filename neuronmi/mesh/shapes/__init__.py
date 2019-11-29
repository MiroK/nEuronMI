from neuronmi.mesh.shapes.taperedneuron import TaperedNeuron
from neuronmi.mesh.shapes.ballstickneuron import BallStickNeuron

from neuronmi.mesh.shapes.microwireprobe import MicrowireProbe
from neuronmi.mesh.shapes.neuropixels24probe import Neuropixels24Probe
from neuronmi.mesh.shapes.neuronexusprobe import NeuronexusProbe

from neuronmi.mesh.shapes.gmsh_primitives import Box, Sphere, Cylinder, Cone

neuron_list = {'bas': BallStickNeuron,
               'tapered': TaperedNeuron}

probe_list = {'microwire': MicrowireProbe,
              'neuronexus': NeuronexusProbe,
              'neuropixels': Neuropixels24Probe}
