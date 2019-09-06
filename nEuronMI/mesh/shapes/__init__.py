from .basneuron import BASNeuron
from .taperedneuron import TaperedNeuron

from .microwireprobe import MicrowireProbe
from .neuronexusprobe import NeuronexusProbe
from .neuropixels24probe import Neuropixels24Probe

neuron_list = {'bas': BASNeuron, 'tapered': TaperedNeuron}
probe_list = {'microwire': MicrowireProbe, 'neuronexus': NeuronexusProbe, 'neuropixels': Neuropixels24Probe}
