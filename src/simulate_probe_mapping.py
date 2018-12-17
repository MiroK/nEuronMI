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
        mesh_path='test.h5'

    mesh_name = os.path.split(mesh_path)[-1]
    assert mesh_name[-3:] == '.h5'

    mesh_root = mesh_name[:-3]
    
    parameters['allow_extrapolation'] = True
    conv = 1E-4
    t_start = time.time()
    mesh, surfaces, volumes, aux_tags = load_mesh(mesh_path)

    elec_dict = probe_contact_map(mesh_path, aux_tags['contact_surfaces'])

    fem_sol_point = join('results/probe_map', mesh_root, 'point', 'u_ext.h5')
    # fem_sol_distr = join('results/probe_map', mesh_root, 'distr', 'u_ext.h5')
    # hdf5_file_distr = HDF5File(mesh.mpi_comm(), fem_sol_distr, "w")
    hdf5_file_point = HDF5File(mesh.mpi_comm(), fem_sol_point, "w")

    for elec, position in elec_dict.items():

        if 'fancy' in mesh_name:
            area = (7.5*conv)**2 * np.pi
        elif 'pixel' in mesh_name:
            area = (12*conv)**2

        problem_params_point = {'cond_ext': 3.0,
                                'stimulated_site': elec,  # or higher by convention
                                'site_current': Expression(('A', '0', '0'), degree=0, A=0,t=0),
                                'point_sources': [position]}

        solver_params = {'dt_fem': 1E-3,  # 1E-3,              # ms
                         'dt_ode': 1E-2,  # 1E-3,               # ms
                         'linear_solver': 'direct'}

        mesh, surfaces, volumes, aux_tags = load_mesh(mesh_path)
        # Where are the probes?
        # ax = plot_contacts(surfaces, aux_tags['contact_surfaces'])

        # s_distr = PoissonSolver(mesh_path=mesh_path,  # Units assuming mesh lengths specified in cm:
        #                         problem_parameters=problem_params_distr,  # ms
        #                         solver_parameters=solver_params)
        s_point = PoissonSolver(mesh_path=mesh_path,  # Units assuming mesh lengths specified in cm:
                                problem_parameters=problem_params_point,  # ms
                                solver_parameters=solver_params)

        # uh_distr = s_distr(None)
        uh_point = s_point([1E-3])
        # print(uh_distr(position - [30*conv,0,0]))
        print(uh_point(position - [30*conv,0,0]))
        print 'Electrode: ' , elec,' Elapsed time: ', time.time() - t_start

        # hdf5_file_distr.write(uh_distr, '/function_%d' % elec)
        hdf5_file_point.write(uh_point, '/function_%d' % elec)
    # hdf5_file_distr.close()
    hdf5_file_point.close()
    np.save(join('results/probe_map', mesh_root, 'point', 'elec_dict'), elec_dict)

