from .solver.neuron_solver import neuron_solver
from .solver.aux import snap_to_nearest
# from .solver.probing import probing_locations
import dolfin
from ..mesh.mesh_utils import EMIEntityMap, load_h5_mesh
import numpy as np
# import yaml

import subprocess, os, time, sys
from os.path import join

def simulate_emi(mesh_folder):
    mesh_folder = Path(mesh_folder)

    mesh_h5 = [f for f in mesh_folder.iterdir() if f.suffix == '.h5']
    mesh_json = [f for f in mesh_folder.iterdir() if f.suffix == '.json']

    if len(mesh_h5) != 1:
        raise ValueError('No or more than one .h5 mesh file found in %s' % mesh_folder)
    else:
        mesh_h5_path = mesh_h5[0]

    if len(mesh_json) != 1:
        raise ValueError('No or more than one .json mesh file found in %s' % mesh_folder)
    else:
        mesh_json_path = mesh_json[0]

    with mesh_json_path.open() as json_fp:
        emi_map = EMIEntityMap(json_fp=json_fp)

    # Current magnitude for probe tip
    # TODO add here stimulation expressions for exp, pulse, step
    magnitude = dolfin.Expression('exp(-1E-2*t)', t=0, degree=1)



    # todo: split problem params in: neurons, external, stimulation
    problem_parameters = {'neuron_0': {'I_ion': dolfin.Constant(0),
                                       'cond': 1,
                                       'C_m': 1,
                                       'stim_strength': 0.0,
                                       'stim_start': 0.0,
                                       'stim_pos': 0.0,
                                       'stim_length': 0.0},
                          #
                          # 'neuron_1': {'I_ion': dolfin.Constant(0),
                          #              'cond': 1,
                          #              'C_m': 1,
                          #              'stim_strength': 0.0,
                          #              'stim_start': 0.0,
                          #              'stim_pos': 0.0,
                          #              'stim_length': 0.0},
                          #
                          'external': {'cond': 2,
                                       'bcs': ('max_x', 'max_y'), },
                          #
                          'probe': {} #'stimulated_sites': ('tip',),
                          #           'site_currents': (magnitude,)}
                          }

    solver_parameters = {'dt_fem': 0.1,
                         'dt_ode': 0.01,
                         'Tstop': 1}

    # TODO extract and save v_mem, v_probe, i_mem
    I_out = dolfin.File(str(mesh_folder / 'I.pvd'))
    for (t, u, I) in neuron_solver(mesh_h5_path, emi_map, problem_parameters, solver_parameters):
        I_out << I, t

#
# if __name__ == '__main__':
#     if '-mesh' in sys.argv:
#         pos = sys.argv.index('-mesh')
#         mesh_path = sys.argv[pos + 1]
#     else:
#         mesh_path='test.h5'
#     if '-probemesh' in sys.argv:
#         pos = sys.argv.index('-probemesh')
#         probe_mesh_path = sys.argv[pos + 1]
#     else:
#         probe_mesh_path=mesh_path
#
#     mesh_name = os.path.split(mesh_path)[-1]
#     assert mesh_name[-3:] == '.h5'
#
#     mesh_root = mesh_name[:-3]
#
#     dolfin.parameters['allow_extrapolation'] = True
#     conv = 1E-4
#     t_start = time.time()
#
#     problem_params = {'C_m': 1.0,    # uF/um^2
#                       'stim_strength': 10.0,             # mS/cm^2
#                       'stim_start': 0.01,                # ms
#                       'stim_pos': 350*conv,              # [0., 0., 350*conv],    # cm
#                       'stim_length': 20*conv,            # cm
#                       'cond_int': 7.0,                   # mS/cm^2
#                       'cond_ext': 3.0,                   # mS/cm^2
#                       'I_ion': 0.0,
#                       'grounded_bottom_only': False,
#                       'Tstop': 5.}                     # ms
#     # Spefication of stimulation consists of 2 parts:
#     # probe tag to be stimulated and the current to be prescribed. The
#     # current has the form normal*A amplitude where normal is INWARD (wrt to probe)
#     # surface normal at the site (assuming the site is flat). This is because we set
#     # bcs on extracellular and use its outward normal. A = A(t) is okay
#     problem_params.update({'stimulated_site': 41,  # or higher by convention
#                            'site_current': Expression(('A', '0', '0'), degree=0, A=1, t=0)})
#
#     solver_params = {'dt_fem': 1E-2, #1E-3,              # ms
#                      'dt_ode': 1E-2, #1E-3,               # ms
#                      'linear_solver': 'direct'}
#
#     mesh, surfaces, volumes, aux_tags = load_mesh(mesh_path)
#     mesh_params = {'path': mesh_path, 'name': mesh_name ,'cells': mesh.num_cells(), 'facets': mesh.num_facets(),
#                    'vertices': mesh.num_vertices(), 'faces': mesh.num_faces(), 'edges': mesh.num_edges(),}
#     performance = {}
#
#     # Solver setup
#     stream = neuron_solver(mesh_path=mesh_path,               # Units assuming mesh lengths specified in cm:
#                            problem_parameters=problem_params,                      # ms
#                            solver_parameters=solver_params)
#
#     if not os.path.isdir('results'):
#         os.mkdir('results')
#
#     if not os.path.isdir(join('results', mesh_root)):
#         os.mkdir(join('results', mesh_root))
#
#     # Get the probes for evary contact surface
#     rec_sites =  np.array(probing_locations(probe_mesh_path, aux_tags['contact_surfaces']))
#
#     # u_file = File(join('results', mesh_root, 'u_sol.pvd'))
#     # I_file = File(join('results', mesh_root, 'current_sol.pvd'))
#
#     u_file = XDMFFile(mpi_comm_world(), join('results', mesh_root, 'u_sol.xdmf'))
#     I_file = XDMFFile(mpi_comm_world(), join('results', mesh_root, 'current_sol.xdmf'))
#
#     # Compute the areas of neuron subdomains for current normalization
#     # NOTE: on the first yield the stream returns the subdomain function
#     # marking the subdomains of the neuron surface
#     neuron_subdomains = next(stream)
#     # Dx here because the mesh is embedded
#     dx_ = Measure('dx', subdomain_data=neuron_subdomains, domain=neuron_subdomains.mesh())
#     areas = {tag: assemble(1 * dx_(tag)) for tag in range(1, 4)}
#
#     I_proxy = None
#     is_neuron_mesh = False
#
#     v_probe = []
#     v_soma = []
#     times = []
#     i_m = []
#
#     p_x, p_y, p_z = rec_sites[0]
#     soma_m = [7.5*conv, 0, 0]
#
#     # Do something with the solutions
#     for n, (t, u, current, system_size) in enumerate(stream):
#
#         if I_proxy is None:
#             I_proxy = snap_to_nearest(current)
#
#         # Store the neuron mesh once for post-processing
#         if not is_neuron_mesh:
#             with HDF5File(mpi_comm_world(), join('results', mesh_root, 'neuron_mesh.h5'), 'w') as nm_out:
#                 nm_out.write(current.function_space().mesh(), 'mesh')
#                 is_neuron_mesh = True
#
#         # msg = 'Normalized curent in %s = %g'
#         # for tag, domain in ((1, 'soma'), (2, 'axon'), (3, 'dendrite')):
#         #     value = assemble(current * dx_(tag))
#         #     value /= areas[tag]
#         #     print msg % (domain, value)
#
#         # print 'At t = %g |u|^2= %g  max(u) = %g min(u) = %g' % (t, u.vector().norm('l2'), u.vector().max(), u.vector().min())
#         print 'Simulation time: ', t , ' v=', u(p_x, p_y, p_z)
#         # print 'I(proxy)=', I_proxy(soma_m[0], soma_m[1], soma_m[2]), \
#         #       'using', I_proxy.snaps[(soma_m[0], soma_m[1], soma_m[2])]
#
#         u_file.write(u, t)
#         I_file.write(current, t)
#         times.append(t)
#         v_probe.append([u(p[0], p[1], p[2]) for p in rec_sites])
#         v_soma.append(u(0, 0, 0))
#         i_m.append(assemble(current * dx_(1))/areas[1])
#
#     t_stop = time.time()
#     processing_time = t_stop - t_start
#     print 'Elapsed time = ', t_stop - t_start
#
#     performance.update({'system size': system_size, 'time': processing_time})
#     v_probe = np.transpose(np.array(v_probe))
#
#     np.save(join('results', mesh_root, 'times'), times)
#     np.save(join('results', mesh_root, 'v_probe'), v_probe)
#     np.save(join('results', mesh_root, 'v_soma'), v_soma)
#     np.save(join('results', mesh_root, 'sites'), rec_sites)
#     np.save(join('results', mesh_root, 'i_soma'), i_m)
#     with open(join('results', mesh_root, 'params.yaml'), 'w') as f:
# 	info = {'problem': problem_params, 'solver': solver_params, 'mesh': mesh_params, 'performance': performance}
# 	yaml.dump(info, f, default_flow_style=False)
#
#     plt.ion()
#     plt.show()

