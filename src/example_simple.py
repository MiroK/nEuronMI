# This example demonstrates the entire pipeline from geometry creation,
# to mesh generation and finally solution. In a realistic workflow
# the 3 steps are done separately.

from mesh.simple_geometry.shapes import SphereNeuron, CylinderProbe
from mesh.simple_geometry.geogen import geofile
from mesh.msh_convert import convert
from solver.neuron_solver import neuron_solver
from solver.aux import snap_to_nearest
from dolfin import *

import subprocess, os, time

h5_is_done = True

if not h5_is_done:
    # Geometry definition
    neuron = SphereNeuron({'rad_soma': 0.5,
                          'rad_dend': 0.3, 'length_dend': 1,
                          'rad_axon': 0.2, 'length_axon': 1,
                          'dxp': 1.5, 'dxn': 1.25, 'dy': 1.0, 'dz': 0.2})
        
    probe = CylinderProbe({'rad_probe': 0.2, 'probe_x': 1.5, 'probe_y': 0, 'probe_z': 0})
                          
    mesh_sizes = {'neuron_mesh_size': 0.2, 'probe_mesh_size': 0.2, 'rest_mesh_size': 0.4}
                          
    # This will give us test.GEO
    geo_file = geofile(neuron, mesh_sizes, probe=probe, file_name='test')
    assert os.path.exists('test.GEO')
                          
    # Generate msh file, test.msh
    subprocess.call(['gmsh -3 test.GEO'], shell=True)
    assert os.path.exists('test.msh')
                          
    # Conversion to h5 file
    convert('test.msh', 'test.h5')
    assert os.path.exists('test.h5')

# Solver setup
stream = neuron_solver(mesh_path='test.h5',               # Units assuming mesh lengths specified in cm:
                       problem_parameters={'C_m': 1.0,    # uF/cm^2
                       'stim_strength': 100.0,            # mS/cm^2
                       'stim_start': 0.1,                 # ms
                       'stim_pos': [0.3, 0.4, 0.3],                   # cm
                       'stim_length': 0.4,                # cm
                       'cond_int': 7.0,                   # mS/cm^2
                       'cond_ext': 3.0,                   # mS/cm^2
                       'I_ion': 0.0,
                       'Tstop': 2.},                     # ms

                       solver_parameters={'dt_fem': 1E-2,#1E-3, # ms
                       'dt_ode': 1E-2,#1E-3,                    # ms
                       'linear_solver': 'direct'})

if not os.path.isdir('results/v_ext'):
    os.mkdir('results')
    os.mkdir('results/v_ext')
    os.mkdir('results/currents')

u_file = File('results/v_ext/u_sol.pvd')
I_file = File('results/currents/current_sol.pvd')

t_start = time.time()
v_probe = []
times = []
i_m = []

# Compute the areas of neuron subdomains for current normalization
# NOTE: on the first yield the stream returns the subdomain function
# marking the subdomains of the neuron surface
neuron_subdomains = next(stream)
# Dx here because the mesh is embedded
dx_ = Measure('dx', subdomain_data=neuron_subdomains, domain=neuron_subdomains.mesh())
areas = {tag: assemble(1*dx_(tag)) for tag in range(1, 4)}
    
I_proxy = None 
# Do something with the solutions
for n, (t, u, current) in enumerate(stream):
    
    if I_proxy is None: I_proxy = snap_to_nearest(current)

    msg = 'Normalized curent in %s = %g'
    for tag, domain in ((1, 'soma'), (2, 'axon'), (3, 'dendrite')):
        value = assemble(current*dx_(tag))
        value /= areas[tag]
        print msg % (domain, value)

    # print 'At t = %g |u|^2= %g  max(u) = %g min(u) = %g' % (t, u.vector().norm('l2'), u.vector().max(), u.vector().min())
    print 'Simulation time: ', t, ' v=', u(1.5, 0, 0)
    print 'I(proxy)=', I_proxy(1.500001, 0.00001, 0.0001), 'using', I_proxy.snaps[(1.500001, 0.00001, 0.0001)]

    if n % 1 == 0:
        u_file << u
        I_file << current
        times.append(t)
        v_probe.append(u(1.5, 0, 0))
        i_m.append(current)

t_stop = time.time()
print 'Elapsed time = ', t_stop - t_start
