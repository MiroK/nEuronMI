from .taperedneuron import TaperedNeuron
from .ballstickneuron import BallStickNeuron

from .microwireprobe import MicrowireProbe

from .gmsh_primitives import Box, Sphere, Cylinder, Cone

neuron_list = {'bas': BallStickNeuron, 'tapered': TaperedNeuron}
probe_list = {'microwire': MicrowireProbe}
