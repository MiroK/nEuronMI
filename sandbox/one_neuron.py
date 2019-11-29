# Sample setup with one neuron
# run as `python one_neuron.py -format msh2`
from neuronmi.mesh.shapes import *
from neuronmi.mesh.mesh_utils import *
import gmsh, sys, json
import numpy as np
from dolfin import File

root = 'test_one_neuron'
# Components
box = Box(np.array([-100, -100, -100]), np.array([200, 200, 400]))
neurons = [BallStickNeuron({'soma_x': 20, 'soma_y': 20, 'soma_z': 0,
                            'soma_rad': 20, 'dend_len': 50, 'axon_len': 50,
                            'dend_rad': 15, 'axon_rad': 10})]


probe = NeuronexusProbe({'tip_x': 100, 'length': 1200, 'angle': np.pi/3})

size_params = {'DistMax': 20, 'DistMin': 10, 'LcMax': 10,
               'neuron_LcMin': 3, 'probe_LcMin': 2}
    
model = gmsh.model
factory = model.occ
# You can pass -clscale 0.25 (to do global refinement)
# or -format msh2            (to control output format of gmsh)
gmsh.initialize(sys.argv)

gmsh.option.setNumber("General.Terminal", 1)

# Add components to model
model, mapping = build_EMI_geometry(model, box, neurons, probe=probe)

# Dump the mapping as json
with open('%s.json' % root, 'w') as out:
    mapping.dump(out)

with open('%s.json' % root) as json_fp:
    mapping = EMIEntityMap(json_fp=json_fp)

# Add fields controlling mesh size
mesh_config_EMI_model(model, mapping, size_params)

factory.synchronize();
# This is a way to store the geometry as geo file
gmsh.write('%s.geo_unrolled' % root)
# Launch gui for visual inspection
gmsh.fltk.initialize()
gmsh.fltk.run()

# From gui you cam make mesh etc ...

gmsh.finalize()
