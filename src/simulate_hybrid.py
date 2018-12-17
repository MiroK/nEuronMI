# This example demonstrates the entire pipeline from geometry creation,
# to mesh generation and finally solution. In a realistic workflow
# the 3 steps are done separately.

from mesh.simple_geometry.shapes import SphereNeuron, CylinderProbe
from mesh.simple_geometry.geogen import geofile
from mesh.msh_convert import convert
from solver.neuron_solver import neuron_solver
from solver.aux import snap_to_nearest
from solver.aux import load_mesh
from solver.probing import probing_locations, plot_contacts, probe_contact_map
from solver.simple_poisson_solver import PoissonSolver
from dolfin import *
import matplotlib.pylab as plt
import numpy as np
import yaml

import subprocess, os, time, sys
from os.path import join

if __name__ == '__main__':
    if '-mesh' in sys.argv:
        pos = sys.argv.index('-mesh')
        mesh_path = sys.argv[pos + 1]
    else:
        mesh_path = 'test.h5'
    if '-cur' in sys.argv:
        pos = sys.argv.index('-cur')
        cur_path = sys.argv[pos + 1]
    else:
        raise Exception()
    if '-pos' in sys.argv:
        pos = sys.argv.index('-pos')
        pos_path = sys.argv[pos + 1]
    else:
        raise Exception()
    if '-fname' in sys.argv:
        fname = sys.argv.index('-fname')
        fname = sys.argv[fname + 1]
    else:
        raise Exception()

    conv = 1E-4
    i_mem = np.loadtxt(cur_path)
    seg_pos = np.loadtxt(pos_path).T
    neuron_path = os.path.dirname(cur_path)
    seg_pos = [list(p*conv) for p in seg_pos]

    mesh_name = os.path.split(mesh_path)[-1]
    assert mesh_name[-3:] == '.h5'

    mesh_root = mesh_name[:-3]

    parameters['allow_extrapolation'] = True
    t_start = time.time()
    mesh, surfaces, volumes, aux_tags = load_mesh(mesh_path)

    elec_dict = probe_contact_map(mesh_path, aux_tags['contact_surfaces'])
    electrode_positions = np.array(elec_dict.values())

    problem_params = {'cond_ext': 3.0,
                      'stimulated_site': 41,  # or higher by convention
                      'site_current': Expression(('A', '0', '0'), degree=0, A=0, t=0),
                      'point_sources': seg_pos
                      }
    solver_params = {'dt_fem': 1E-3,  # 1E-3,              # ms
                     'dt_ode': 1E-2,  # 1E-3,               # ms
                     'linear_solver': 'direct'}

    mesh, surfaces, volumes, aux_tags = load_mesh(mesh_path)
    mesh_params = {'path': mesh_path, 'name': mesh_name, 'cells': mesh.num_cells(), 'facets': mesh.num_facets(),
                   'vertices': mesh.num_vertices(), 'faces': mesh.num_faces(), 'edges': mesh.num_edges()}
    performance = {}
    # Where are the probes?
    # ax = plot_contacts(surfaces, aux_tags['contact_surfaces'])
    s = PoissonSolver(mesh_path=mesh_path,  # Units assuming mesh lengths specified in cm:
                      problem_parameters=problem_params,  # ms
                      solver_parameters=solver_params)
    v_ext = np.zeros((len(electrode_positions), i_mem.shape[1]))
    t_start = time.time()
    for i, im in  enumerate(i_mem.T):
        uh = s(list(im))
        print 'Timestep ', i+1, ' of ', i_mem.shape[1]
        print 'Elapsed time: ', time.time() - t_start
        v_ext[:, i] = np.array([uh(p) for p in electrode_positions])

    processing_time = time.time() - t_start
    performance.update({'system size': s.system_size, 'time': processing_time})

    np.savetxt(join(neuron_path, fname +'.txt'), v_ext)
    with open(join(neuron_path, 'params.yaml'), 'w') as f:
        info = {'solver': solver_params, 'mesh': mesh_params, 'performance': performance}
        yaml.dump(info, f, default_flow_style=False)