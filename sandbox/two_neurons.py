from neuronmi.mesh.shapes import *
from neuronmi.mesh.mesh_utils import *
import gmsh, sys, json, os
import numpy as np
from dolfin import File, FunctionSpace, FiniteElement, MixedElement


def make_coarse_mesh():
    root = 'test_2neuron'
    msh_file = '%s.msh' % root
    json_file = '%s.json' % root

    # This gives course enough mesh that the solver runs fast
    box = Box(np.array([-60, -60, -100]), np.array([60, 60, 100]))

    neurons = [BallStickNeuron({'soma_x': 0, 'soma_y': 0, 'soma_z': 0,
                                'soma_rad': 20, 'dend_len': 50, 'axon_len': 60,
                                'dend_rad': 15, 'axon_rad': 10}),
               TaperedNeuron({'soma_x': 30, 'soma_y': -30, 'soma_z': 0,
                              'soma_rad': 20, 'dend_len': 20, 'axon_len': 50, 'axonh_len': 40, 'dendh_len': 11,
                              'dend_rad': 10, 'axon_rad': 8, 'axonh_rad': 10, 'dendh_rad': 15})]

    probe = MicrowireProbe({'tip_x': 30, 'radius': 5, 'length': 800})

    # Coarse enough for tests
    size_params = {'DistMax': 20, 'DistMin': 10, 'LcMax': 40,
                   'neuron_LcMin': 6, 'probe_LcMin': 6}
    
    # You can pass -clscale 0.25 (to do global refinement)
    gmsh.initialize(sys.argv)
    # Force compatibility with dolfin
    gmsh.option.setNumber('Mesh.PreserveNumberingMsh2', 1)
    gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)
    
    model = gmsh.model
    factory = model.occ
    
    gmsh.option.setNumber("General.Terminal", 1)

    # Add components to model
    model, mapping = build_EMI_geometry(model, box, neurons, probe=probe)
    
    with open(json_file, 'w') as out:
        mapping.dump(out)

    # Dump the mapping as json
    mesh_config_EMI_model(model, mapping, size_params)
    factory.synchronize()
    
    # gmsh.fltk.initialize()
    # gmsh.fltk.run()
    
    # This is a way to store the geometry as geo file
    gmsh.write('%s.geo_unrolled' % root)
    # 3d model
    model.mesh.generate(3)
    # Native optimization
    model.mesh.optimize('')
    gmsh.write(msh_file)
    gmsh.finalize()

    # Convert
    h5_file = msh_to_h5(msh_file)
    mesh, volumes, surfaces = load_h5_mesh(h5_file)

    # Eye check
    File('%s_surfaces.pvd' % root) << surfaces
    File('%s_volumes.pvd' % root) << volumes

    # Sanity checks
    h5_file = msh_to_h5(msh_file)
    assert os.path.exists(h5_file)
    
    mesh, volumes, surfaces = load_h5_mesh(h5_file)

    volumes = set(volumes.array())
    # We have three volumes: inside1, inside2 and outside neuron
    assert volumes == set((1, 2, 3))
    
    surfaces0 = set(surfaces.array())

    objects = ('box', 'probe', 'neuron_0', 'neuron_1')
    surfaces = set(sum((mapping.surface_physical_tags(k).values() for k in objects), []))
    # The remaing unmarked facets are interior 0 
    assert (surfaces0 - surfaces) == set((0, ))
        
    # Space size for PDE?
    cell = mesh.ufl_cell()
    Welm = MixedElement([FiniteElement('Raviart-Thomas', cell, 1),
                         FiniteElement('Discontinuous Lagrange', cell, 0),
                         FiniteElement('Discontinuous Lagrange Trace', cell, 0)])
    W = FunctionSpace(mesh, Welm)
    
    print('EMI PDE dim(W)', W.dim())

    return True

# --------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    try:
        success = make_coarse_mesh()
    except:
        success = False

    sys.exit(int(success))
