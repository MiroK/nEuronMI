# This example demonstrates the entire pipeline from geometry creation,
# to mesh generation and finally solution. In a realistic workflow
# the 3 steps are done separately.

from mesh.simple_geometry.shapes import SphereNeuron, CylinderProbe
from mesh.simple_geometry.geogen import geofile
from mesh.msh_convert import convert
from solver.neuron_solver import neuron_solver
from solver.probing import probing_locations
from dolfin import *
import matplotlib.pylab as plt
import numpy as np

import subprocess, os, time, sys
from os.path import join


if __name__ == '__main__':
    if '-mesh' in sys.argv:
        pos = sys.argv.index('-mesh')
        mesh_path = sys.argv[pos + 1]
    else:
        mesh_path='test.h5'
    if '-probemesh' in sys.argv:
        pos = sys.argv.index('-probemesh')
        probe_mesh_path = sys.argv[pos + 1]
    else:
        probe_mesh_path=mesh_path

    mesh_name = os.path.split(mesh_path)[-1]
    assert mesh_name[-3:] == '.h5'

    mesh_root = mesh_name[:-3]

    conv = 1E-4
    t_start = time.time()

    # Solver setup
    stream = neuron_solver(mesh_path=mesh_path,               # Units assuming mesh lengths specified in cm:
                           problem_parameters={'C_m': 1.0,    # uF/um^2
                           'stim_strength': 10.0,             # mS/cm^2
                           'stim_start': 0.01,                # ms
                           'stim_pos': [0., 0., 350*conv],    # cm
                           'stim_length': 20*conv,            # cm
                           'cond_int': 7.0,                   # mS/cm^2
                           'cond_ext': 3.0,                   # mS/cm^2
                           'I_ion': 0.0,
                           'Tstop': 5.},                      # ms
                           solver_parameters={'dt_fem': 1E-2, #1E-3, # ms
                           'dt_ode': 1E-2,#1E-3,                    # ms
                           'linear_solver': 'direct'})

    if not os.path.isdir('results'):
        os.mkdir('results')

    if not os.path.isdir(join('results', mesh_root)):
        os.mkdir(join('results', mesh_root))

    rec_sites =  np.array(probing_locations(probe_mesh_path, 41))

    u_file = File(join('results', mesh_root, 'u_sol.pvd'))
    I_file = File(join('results', mesh_root, 'current_sol.pvd'))

    v_probe = []
    times = []
    i_m = []

    p_x, p_y, p_z = rec_sites[0]

    # Do something with the solutions
    for n, (t, u, current) in enumerate(stream):
        # print 'At t = %g |u|^2= %g  max(u) = %g min(u) = %g' % (t, u.vector().norm('l2'), u.vector().max(), u.vector().min())
        print 'Simulation time: ', t , ' v=', u(p_x, p_y, p_z)

        # if n % 50 == 0:
        u_file << u
        I_file << current

        times.append(t)
        v_probe.append([u(p[0], p[1], p[2]) for p in rec_sites])


    t_stop = time.time()
    print 'Elapsed time = ', t_stop - t_start

    v_probe = np.transpose(np.array(v_probe))

    np.save(join('results', mesh_root, 'times'), times)
    np.save(join('results', mesh_root, 'v_probe'), v_probe)
    np.save(join('results', mesh_root, 'sites'), rec_sites)

    plt.ion()
    plt.show()

