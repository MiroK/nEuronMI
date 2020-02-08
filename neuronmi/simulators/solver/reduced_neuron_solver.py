from neuronmi.simulators.solver.aux import SiteCurrent, surface_normal
from neuronmi.simulators.solver.linear_algebra import LinearSystemSolver
from neuronmi.simulators.solver.transferring import SubMeshTransfer
from neuronmi.simulators.solver.membrane import MembraneODESolver
from neuronmi.mesh.mesh_utils import load_h5_mesh
import numpy as np
import itertools

from dolfin import *

from neuronmi.simulators.solver.union import UnionMesh, UnionFunction
import xii.assembler.average_matrix
from xii.meshing.transfer_markers import transfer_markers
from block.block_bc import block_rhs_bc
from block.algebraic.petsc import LU
from xii import *



def neuron_solver(mesh_path, emi_map, problem_parameters, scale_factor=None, verbose=False):
    '''
    Solver for the Reduced Primal Single-dimensional formulation of the EMI equations
    
    mesh_path: str that is the path to HDF5File containing mesh, ...
    emi_map: EMIEntityMap of the mesh
    problem_parameters: dict specifying the following

      For each neuron (neuron_i) the I_ion, cond[uctivity], Cm, parameters for stim[_*]ulation params
      For exterior (external) cond[uctivity], names of insulated exterior boundaries
      For probe stimulated_sites (named) and their currents

    solver_parameters: time_step, dt (of EMI), dt_ode
    '''
    mesh_path = str(mesh_path)
    mesh, volume_marking_f, facet_marking_f, edge_marking_f = load_h5_mesh(mesh_path, scale_factor)

    num_neurons = emi_map.num_neurons
    assert num_neurons == 1, emi_map.num_neurons

    solver_parameters = problem_parameters['solver']
    neurons_parameters = problem_parameters['neurons']
    ext_parameters = problem_parameters['ext']
    probe_parameters = problem_parameters['probe']

    # TODO use v_rest to initialize intracellular potential initial condition
    v_rest = -75
    I_ion = Constant(0)

    # Do we have properties for each one
    if isinstance(neurons_parameters, list):
        assert len(
            neurons_parameters) == num_neurons, "If 'neurons' parameter is a list, the lentgh must be the same as" \
                                                "the number of neurons in the mesh"
    else:
        neurons_parameters = [neurons_parameters] * num_neurons

    # There is only one 3d mesh
    ext_Vtag = emi_map.volume_physical_tags('external')['all']
    assert len(emi_map.volumes) == 1
    
    # Neurons as 1d meshes
    meshes_neuron = [EmbeddedMesh(edge_marking_f, emi_map.curve_physical_tags('neuron_%d' % i).values())
                      for i in range(num_neurons)]
    # Let's create a cell function which tags each cell of neuron as soma, dendrite, etc ...
    # And a P0 function which has the radius
    neurons_tags, neurons_radii = [], []
    for i, neuron_mesh in enumerate(meshes_neuron):
        ctypes = emi_map.curve_types('neuron_%d' % i)
        # Each cell in marking function has some physical tag
        cell_f = MeshFunction('size_t', neuron_mesh, neuron_mesh.topology().dim(), 0)
        # That physical tag is used to look up the type of neuron
        values = np.array([ctypes[neuron_mesh.marking_function[ci]] for ci in range(neuron_mesh.num_cells())],
                          dtype='uintp')
        cell_f.array()[:] = values
        # Add
        neurons_tags.append(cell_f)

        radii = emi_map.curve_radii('neuron_%d' % i)
        values = np.array([radii[neuron_mesh.marking_function[ci]] for ci in range(neuron_mesh.num_cells())],
                          dtype=float)
        # Unit conversion
        values *= scale_factor
        
        # NOTE: ignore DG0 dofmap
        Q = FunctionSpace(neuron_mesh, 'DG', 0)
        radius = Function(Q)
        radius.vector().set_local(values)
        # Add
        neurons_radii.append(radius)
        
    # We solve for extrac. potential in the full mesh
    # and intracellular potentials on the lines
    Ve = FunctionSpace(mesh, 'CG', 1)
    Vis = [FunctionSpace(neuron_mesh, 'CG', 1) for neuron_mesh in meshes_neuron]
    # The total space is
    W = [Ve] + Vis
    print 'W dim is', sum(Wi.dim() for Wi in W)

    # Build the block operator
    ue, uis = TrialFunction(Ve), map(TrialFunction, Vis)
    ve, vis = TestFunction(Ve), map(TestFunction, Vis)
    # We will need to reduce 3d extracellular potential to line
    Tues = [Average(ue, neuron, Circle(radius, degree=10))
            for neuron, radius in zip(meshes_neuron, neurons_radii)]

    Tves = [Average(ve, neuron, Circle(radius, degree=10))
            for neuron, radius in zip(meshes_neuron, neurons_radii)]
    
    # And with each surface we need to associate a measure
    dx_ = [Measure('dx', domain=neuron) for neuron in meshes_neuron]
    # For setting up Nuemann bcs on extrac. potential we need
    ds_ = Measure('ds', domain=mesh, subdomain_data=facet_marking_f)

    dt_fem = Constant(solver_parameters['dt_fem'])    
    # Now the block operators
    a = block_form(W, 2)
    # Laplacians - external
    a[0][0] = Constant(ext_parameters['cond_ext'])*inner(grad(ue), grad(ve))*dx
    # Coupling contrib to diagonal of external
    for i, (Tue_i, Tve_i, dx_i) in enumerate(zip(Tues, Tves, dx_)):
        scale = Constant(neurons_parameters[i]['Cm']/dt_fem)
        rad = neurons_radii[i]
        a[0][0] += scale*2*pi*rad*inner(Tue_i, Tve_i)*dx_i
    # Internal
    for i, (ui, vi, dx_i) in enumerate(zip(uis, vis, dx_), 1):
        cond = Constant(neurons_parameters[i-1]['cond_int'])
        scale = Constant(neurons_parameters[i-1]['Cm']/dt_fem)
        rad = neurons_radii[i-1]

        a[i][i] = (cond*pi*rad**2*inner(grad(ui), grad(vi))*dx +
                   scale*2*pi*rad*inner(ui, vi)*dx_i)

    # Offdiagonal 
    for i, (Tve_i, ui, dx_i) in enumerate(zip(Tves, uis, dx_), 1):
        scale = Constant(neurons_parameters[i-1]['Cm']/dt_fem)
        rad = neurons_radii[i-1]        
        a[0][i] = -2*pi*rad*scale*inner(Tve_i, ui)*dx_i

    for i, (Tue_i, vi, dx_i) in enumerate(zip(Tues, vis, dx_), 1):
        scale = Constant(neurons_parameters[i-1]['Cm']/dt_fem)
        rad = neurons_radii[i-1]                
        a[i][0] = -2*pi*rad*scale*inner(Tue_i, vi)*dx_i
        
    # Boundary conditions: grounded surfaces are Dirichlet
    #                      neumann surfaces are part of weak form and
    #                       - insulation means the integral is zero so
    #                         we only care about stimulated    
    grounded_tags = set(emi_map.surface_physical_tags('box').values())  # All
    grounded_tags.difference_update(set(emi_map.surface_physical_tags('box')[name]
                                        for name in ext_parameters['insulated_bcs']))
    
    # Bcs to be put explicitely on system
    Ve_bcs = [DirichletBC(Ve, Constant(0), facet_marking_f, tag)
              for tag in grounded_tags]
    # No other subspace has bcs; Vis, Qis
    W_bcs = [Ve_bcs] + [list() for _ in range(num_neurons)]

    # Right hand side is formed by Neuumann nonzero terms on stimulated
    stimulated_map = {}
    # Add the stimulated site
    if 'probe' in emi_map.surfaces:
        probe_surfaces = emi_map.surface_physical_tags('probe')  # Dict
        # Stimulated sites must be a list of contact_names
        if 'stimulated_sites' in probe_parameters.keys():
            if probe_parameters['stimulated_sites'] is not None:
                if len(probe_parameters['stimulated_sites']) > 0:
                    site_currents = probe_parameters['current']

                    if isinstance(site_currents, (int, float)):
                        site_currents = [site_currents] * len(probe_parameters['stimulated_sites'])
                    else:
                        assert len(site_currents) == len(probe_parameters['stimulated_sites']), "Length of probe " \
                                                                                                "'currents' and " \
                                                                                                "'stimulated_sites' " \
                                                                                                "should correspond"
                    for name, current in zip(probe_parameters['stimulated_sites'], site_currents):
                        tag = probe_surfaces[name]
                        stimulated_map[tag] = current

    # And the time stepping for the current; Let p0i be a transmemebrane
    # potential on the i-th neuron
    Qis = [FunctionSpace(mesh_neuron, 'DG', 0) for mesh_neuron in meshes_neuron]
    p0is = [interpolate(Constant(v_rest), Qi) for Qi in Qis]
    # Membrane current; we want to update these using C_m*dv/dt = I_m
    Ims = [Function(Qi) for Qi in Qis]
    
    L = block_form(W, 1)
    L[0] = inner(Constant(0), ve)*dx + sum(inner(current, ve)*ds_(tag) for tag, current in stimulated_map.items())
    for i, (p0i, Tve_i, vi, dx_i) in enumerate(zip(p0is, Tves, vis, dx_), 1):
        scale = Constant(neurons_parameters[i-1]['Cm']/dt_fem)
        rad = neurons_radii[i-1]        
        
        L[0] += -scale*2*pi*rad*inner(p0i, Tve_i)*dx_i
        L[i] = scale*2*pi*rad*inner(p0i, vi)*dx_i
    
    # FIXME: setup the solver here
    A, b = map(ii_assemble, (a, L))
    # Apply bcs to the system
    A_bc, _ = apply_bc(A, b, W_bcs)
    b_bc = block_rhs_bc(W_bcs, A)
    # Solver is setup based on monolithic
    A_mono = ii_convert(A_bc)
    print('Setting up solver')
    time_Ainv = Timer('Ainv')
    # A_inv = LU(A_mono)  # Setup once
    A_inv = PETScLUSolver(A_mono, 'umfpack')
    print('Done in %g s' % time_Ainv.stop())
    
    dt_ode = solver_parameters['dt_ode']
    assert dt_ode <= dt_fem(0)
    # Setup neuron
    fem_ode_sync = int(dt_fem(0) / dt_ode)

    neuron_solutions = []
    for i, neuron in enumerate(meshes_neuron):
        # Pick the nueuron from neuron collection
        ode_solver = MembraneODESolver(neurons_tags[i],
                                       # SWC
                                       soma=(1, ), axon=(2,) if 2 in neurons_tags[i] else (), dendrite=(3, ),
                                       problem_parameters=neurons_parameters[i],
                                       scale_factor=scale_factor,
                                       # In mortar we want ODE in vertices
                                       solver_parameters={'V_polynomial_family': 'Discontinuous Lagrange',
                                                          'V_polynomial_degree': 0,
                                                          'S_polynomial_family': 'Discontinuous Lagrange',
                                                          'S_polynomial_degree': 0})

        sim_duration = solver_parameters['sim_duration']
        assert sim_duration > 0.0
        interval = (0.0, sim_duration)

        # NOTE: a generator; nothing is computed so far
        ode_solutions = ode_solver.solve(interval, dt_ode)  # Potentials only
        neuron_solutions.append(ode_solutions)

    get_transmembrane_potentials = []
    for i, (Tvei, ui, p0i) in enumerate(zip(Tves, uis, p0is)):
        # -Pi u_e + u_i; only ext is restricited
        Qi = p0i.function_space()
        #average_matrix(Ve, Qi, Tvei.average_['shape'])        
        ext_mat = PETScMatrix(average_matrix.average_matrix(Ve, Qi, Tvei.average_['shape']))
        # However we restrict to P0; this is also the space of p0i but not
        # Vi - so we project
        q = TestFunction(Qi)
        Minv_b = lambda ui, q=q: assemble((1/CellVolume(q.function_space().mesh()))*inner(ui, q)*dx)

        get_transmembrane_potentials.append(
            lambda ue, ui, Te=ext_mat, Minv_b=Minv_b: -Te*ue.vector() + Minv_b(ui)
        )

    # Solution in Ve x Vi x Q
    wh = ii_Function(W)
    # Set initial intracellular; we don't know about currents
    for i in range(1, 1+num_neurons):
        wh[i].vector().zero()
        wh[i].vector()[:] += v_rest  # Set resting potential

    # To get initial state
    yield 0, wh

    neuron_solutions = itertools.izip(*neuron_solutions)

    step_count = 0
    for odes in neuron_solutions:
        step_count += 1

        (t0, t1) = odes[0][0]
        print('Time is (%g, %g)' % (t0, t1))
        if step_count == fem_ode_sync:
            step_count = 0
            # Set transmemebrane potentials for PDE based on ODE
            for i in range(num_neurons):
                p0is[i].vector().zero()
                p0is[i].vector().axpy(1, odes[i][1].vector())

            # We could have changing in time simulation
            for _, I in stimulated_map.values():  
                if 't' in I:
                    I.t = float(t1)

            # FIXME: Reassemble
            #aux, b = map(ii_assemble, (a, L))
            #_, b = apply_bc(aux, b, W_bcs)  # Apply updated bcs

            b = ii_assemble(L)
            b_bc.apply(b)
            
            # New solution
            if verbose:
                print('\tSolving linear system of size %d' % A.size(0))
            # Solve with new rhs
            A_inv.solve(wh.vector(), ii_convert(b))
           #  wh.vector()[:] = A_inv*ii_convert(b)  # With monolithic

            # Update transembrane potential
            uh_e = wh[0]
            uh_is = wh[1:(num_neurons+1)]
            for i, (p0i, uh_i, get_vi) in enumerate(zip(p0is, uh_is, get_transmembrane_potentials), 1):
                p0i.vector().zero()
                p0i.vector().axpy(1, get_vi(uh_e, uh_i))

            e = lambda x: sqrt(abs(assemble(inner(x, x)*dx)))
            print e(uh_e), [e(uh_i) for uh_i in uh_is]
                
            yield t1, wh

            # Get transmembrane potential to ODE for next round
            for i in range(num_neurons):
                odes[i][1].vector().zero()
                odes[i][1].vector().axpy(1, p0is[i].vector())
