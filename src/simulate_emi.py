# This example demonstrates the entire pipeline from geometry creation,
# to mesh generation and finally solution. In a realistic workflow
# the 3 steps are done separately.

from mesh.simple_geometry.shapes import SphereNeuron, CylinderProbe
from mesh.simple_geometry.geogen import geofile
from mesh.msh_convert import convert
from solver.neuron_solver import neuron_solver
from solver.aux import snap_to_nearest
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
                           'stim_pos': 350*conv, #[0., 0., 350*conv],    # cm
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

    # u_file = File(join('results', mesh_root, 'u_sol.pvd'))
    # I_file = File(join('results', mesh_root, 'current_sol.pvd'))

    u_file = XDMFFile(mpi_comm_world(), 'results/v_ext/u_sol.xdmf')
    I_file = XDMFFile(mpi_comm_world(), 'results/currents/current_sol.xdmf')

    # Compute the areas of neuron subdomains for current normalization
    # NOTE: on the first yield the stream returns the subdomain function
    # marking the subdomains of the neuron surface
    neuron_subdomains = next(stream)
    # Dx here because the mesh is embedded
    dx_ = Measure('dx', subdomain_data=neuron_subdomains, domain=neuron_subdomains.mesh())
    areas = {tag: assemble(1 * dx_(tag)) for tag in range(1, 4)}

    I_proxy = None

    v_probe = []
    times = []
    i_m = []

    p_x, p_y, p_z = rec_sites[0]
    soma_m = [15*conv, 0, 0]

    # Do something with the solutions
    for n, (t, u, current) in enumerate(stream):

        if I_proxy is None: I_proxy = snap_to_nearest(current)

        msg = 'Normalized curent in %s = %g'
        for tag, domain in ((1, 'soma'), (2, 'axon'), (3, 'dendrite')):
            value = assemble(current * dx_(tag))
            value /= areas[tag]
            print msg % (domain, value)

        # print 'At t = %g |u|^2= %g  max(u) = %g min(u) = %g' % (t, u.vector().norm('l2'), u.vector().max(), u.vector().min())
        print 'Simulation time: ', t, ' v=', u(p_x, p_y, p_z)
        print 'I(proxy)=', I_proxy(soma_m[0], soma_m[1], soma_m[2]), \
            'using', I_proxy.snaps[(soma_m[0], soma_m[1], soma_m[2])]

        if n % 1 == 0:
            u_file.write(u, t)
            I_file.write(current, t)

            times.append(t)
            v_probe.append(u(p_x, p_y, p_z))
            i_m.append(current)

    u_file.close()
    I_file.close()

    t_stop = time.time()
    print 'Elapsed time = ', t_stop - t_start

    v_probe = np.transpose(np.array(v_probe))

    np.save(join('results', mesh_root, 'times'), times)
    np.save(join('results', mesh_root, 'v_probe'), v_probe)
    np.save(join('results', mesh_root, 'sites'), rec_sites)
    np.save(join('results', mesh_root, 'i_soma'), i_m)

    plt.ion()
    plt.show()

