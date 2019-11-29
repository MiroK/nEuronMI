# Run as
# python generation.py -format msh2
# Generate mesh for box with probe - we find all the contacts

from neuronmi.mesh.shapes import *
from neuronmi.mesh.mesh_utils import *
from neuronmi.simulators.solver.probing import get_geom_centers, Probe
from dolfin import FunctionSpace, interpolate, Expression, File
import gmsh, sys, json
import numpy as np

root = 'test_contacts'
# Components
box = Box(np.array([-100, -100, -100]), np.array([200, 200, 400]))
# NOTE: The trick we play below is that we evaluate field x. For it to
# work as an assert the probe cannot be rotated
probe = NeuronexusProbe({'tip_x': 100, 'length': 1200})

size_params = {'DistMax': 20, 'DistMin': 10, 'LcMax': 10,
               'neuron_LcMin': 3, 'probe_LcMin': 4}

model = gmsh.model
factory = model.occ
gmsh.initialize(sys.argv)

gmsh.option.setNumber("General.Terminal", 1)

# Add components to model
model, mapping = build_EMI_geometry(model, box, neurons=None, probe=probe)

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
# gmsh.fltk.initialize()
# gmsh.fltk.run()

# 3d model
model.mesh.generate(3)
# Native optimization
model.mesh.optimize('')
gmsh.write('test.msh')
# Convert
h5_file = msh_to_h5('test.msh')

mesh, volumes, surfaces = load_h5_mesh('test.h5')

# Eye check
File('surfaces.pvd') << surfaces
File('volumes.pvd') << volumes

# Contact check
probe_surfaces = mapping.surface_physical_tags('probe')
contact_tags = [v for k, v in probe_surfaces.items() if 'contact_' in k]

contact_centers = get_geom_centers(surfaces, contact_tags)
contact_centers = np.array(contact_centers)

V = FunctionSpace(mesh, 'CG', 1)
f = interpolate(Expression('x[0]', degree=1), V)

probe = Probe(f, locations=contact_centers)
# Sample
probe.probe(0)
data_0 = probe.data[0]
# First is time; don't care
contact_x0 = data_0[1:]
# The truth
contact_x = np.array(contact_centers)[:, 0]
assert np.linalg.norm(contact_x - contact_x0, np.inf) < 1E-13
