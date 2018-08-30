# This example demonstrates the entire pipeline from geometry creation,
# to mesh generation and finally solution. In a realistic workflow
# the 3 steps are done separately.

from mesh.simple_geometry.shapes import SphereNeuron, CylinderProbe
from mesh.simple_geometry.geogen import geofile
from mesh.msh_convert import convert
from solver.neuron_solver import neuron_solver
from solver.aux import snap_to_nearest
from solver.aux import load_mesh
from solver.probing import probing_locations
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
    if '-probemesh' in sys.argv:
        pos = sys.argv.index('-probemesh')
        probe_mesh_path = sys.argv[pos + 1]
    else:
        probe_mesh_path=mesh_path

    mesh_name = os.path.split(mesh_path)[-1]
    assert mesh_name[-3:] == '.h5'

    mesh_root = mesh_name[:-3]
    
    parameters['allow_extrapolation'] = True
    conv = 1E-4
    t_start = time.time()

    # if 'mainen2' in mesh_name or 'sphere2' in mesh_name:
    other_neuron = 99
    # else:
    #     other_neuron = None

    problem_params = {'C_m': 1.0,    # uF/um^2
                      'stim_strength': 10.0,             # mS/cm^2
                      'stim_start': 0.01,                # ms
                      'stim_pos': 350*conv, #[0., 0., 350*conv],    # cm
                      'stim_length': 20*conv,            # cm
                      'cond_int': 7.0,                   # mS/cm^2
                      'cond_ext': 3.0,                   # mS/cm^2
                      'I_ion': 0.0,
                      'grounded_bottom_only': False,
                      'Tstop': 5.,
                      'other_neuron': other_neuron}                       # ms
    solver_params = {'dt_fem': 1E-2, #1E-3,              # ms
                     'dt_ode': 1E-2, #1E-3,               # ms
                     'linear_solver': 'direct'}

    mesh, surfaces, volumes, aux_tags = load_mesh(mesh_path, other_neuron=problem_params['other_neuron'])
    mesh_params = {'path': mesh_path, 'name': mesh_name ,'cells': mesh.num_cells(), 'facets': mesh.num_facets(),
                   'vertices': mesh.num_vertices(), 'faces': mesh.num_faces(), 'edges': mesh.num_edges(),}
    performance = {}

    # Solver setup
    stream = neuron_solver(mesh_path=mesh_path,               # Units assuming mesh lengths specified in cm:
                           problem_parameters=problem_params,                      # ms
                           solver_parameters=solver_params)

    if not os.path.isdir('results'):
        os.mkdir('results')

    if not os.path.isdir(join('results', mesh_root)):
        os.mkdir(join('results', mesh_root))

    rec_sites =  np.array(probing_locations(probe_mesh_path, 41, problem_params['other_neuron']))

    # u_file = File(join('results', mesh_root, 'u_sol.pvd'))
    # I_file = File(join('results', mesh_root, 'current_sol.pvd'))

    u_file = XDMFFile(mpi_comm_world(), join('results', mesh_root, 'u_sol.xdmf'))
    I_file = XDMFFile(mpi_comm_world(), join('results', mesh_root, 'current_sol.xdmf'))

    # Compute the areas of neuron subdomains for current normalization
    # NOTE: on the first yield the stream returns the subdomain function
    # marking the subdomains of the neuron surface
    neuron_subdomains = next(stream)
    # Dx here because the mesh is embedded
    dx_ = Measure('dx', subdomain_data=neuron_subdomains, domain=neuron_subdomains.mesh())
    areas = {tag: assemble(1 * dx_(tag)) for tag in range(1, 4)}

    I_proxy = None
    is_neuron_mesh = False

    v_probe = []
    v_soma = []
    times = []
    i_m = []

    p_x, p_y, p_z = rec_sites[0]
    # p_x, p_y, p_z = np.array([200*conv, 0, 0])
    soma_m = [7.5*conv, 0, 0]

    # Do something with the solutions
    for n, (t, u, current, system_size) in enumerate(stream):

        if I_proxy is None:
            I_proxy = snap_to_nearest(current)

        # Store the neuron mesh once for post-processing
        if not is_neuron_mesh:
            with HDF5File(mpi_comm_world(), join('results', mesh_root, 'neuron_mesh.h5'), 'w') as nm_out:
                nm_out.write(current.function_space().mesh(), 'mesh')
                is_neuron_mesh = True

        # msg = 'Normalized curent in %s = %g'
        # for tag, domain in ((1, 'soma'), (2, 'axon'), (3, 'dendrite')):
        #     value = assemble(current * dx_(tag))
        #     value /= areas[tag]
        #     print msg % (domain, value)

        # print 'At t = %g |u|^2= %g  max(u) = %g min(u) = %g' % (t, u.vector().norm('l2'), u.vector().max(), u.vector().min())
        print 'Simulation time: ', t , ' v=', u(p_x, p_y, p_z)
        # print 'I(proxy)=', I_proxy(soma_m[0], soma_m[1], soma_m[2]), \
        #       'using', I_proxy.snaps[(soma_m[0], soma_m[1], soma_m[2])]

        u_file.write(u, t)
        I_file.write(current, t)
        times.append(t)
        v_probe.append([u(p[0], p[1], p[2]) for p in rec_sites])
        v_soma.append(u(0, 0, 0))
        i_m.append(assemble(current * dx_(1))/areas[1])

    t_stop = time.time()
    processing_time = t_stop - t_start
    print 'Elapsed time = ', t_stop - t_start

    performance.update({'system size': system_size, 'time': processing_time})
    v_probe = np.transpose(np.array(v_probe))

    np.save(join('results', mesh_root, 'times'), times)
    np.save(join('results', mesh_root, 'v_probe'), v_probe)
    np.save(join('results', mesh_root, 'v_soma'), v_soma) 
    np.save(join('results', mesh_root, 'sites'), rec_sites)
    np.save(join('results', mesh_root, 'i_soma'), i_m)
    with open(join('results', mesh_root, 'params.yaml'), 'w') as f:
	info = {'problem': problem_params, 'solver': solver_params, 'mesh': mesh_params, 'performance': performance}
	yaml.dump(info, f, default_flow_style=False)
        
    plt.ion()
    plt.show()

