from neuronmi.simulators.solver.aux import SiteCurrent, surface_normal
from neuronmi.simulators.solver.linear_algebra import LinearSystemSolver
from neuronmi.simulators.solver.transferring import SubMeshTransfer
from neuronmi.simulators.solver.membrane import MembraneODESolver
from neuronmi.mesh.mesh_utils import load_h5_mesh
import numpy as np
import itertools

from dolfin import *

from neuronmi.simulators.solver.union import UnionMesh, UnionFunction
from xii.assembler.trace_matrix import trace_mat_no_restrict
from xii.meshing.transfer_markers import transfer_markers
from block.block_bc import block_rhs_bc
from block.algebraic.petsc import LU
from xii import *


def neuron_solver(mesh_path, emi_map, problem_parameters, scale_factor=None, verbose=False):
    '''
    Solver for the Primal multiscale formulation of the EMI equations
    
    mesh_path: str that is the path to HDF5File containing mesh, ...
    emi_map: EMIEntityMap of the mesh
    problem_parameters: dict specifying the following

      For each neuron (neuron_i) the I_ion, cond[uctivity], Cm, parameters for stim[_*]ulation params
      For exterior (external) cond[uctivity], names of insulated exterior boundaries
      For probe stimulated_sites (named) and their currents

    solver_parameters: time_step, dt (of EMI), dt_ode
    '''
    mesh_path = str(mesh_path)
    mesh, volume_marking_f, facet_marking_f = load_h5_mesh(mesh_path, scale_factor)

    solver_parameters = problem_parameters['solver']
    neurons_parameters = problem_parameters['neurons']
    ext_parameters = problem_parameters['ext']
    probe_parameters = problem_parameters['probe']

    # TODO use v_rest to initialize intracellular potential initial condition
    v_rest = -75
    I_ion = Constant(0)

    num_neurons = emi_map.num_neurons
    # Do we have properties for each one
    if isinstance(neurons_parameters, list):
        assert len(
            neurons_parameters) == num_neurons, "If 'neurons' parameter is a list, the lentgh must be the same as" \
                                                "the number of neurons in the mesh"
    else:
        neurons_parameters = [neurons_parameters] * num_neurons

    # Define subspaces for ext and neurons
    ext_Vtag = emi_map.volume_physical_tags('external')['all']
    n_Vtags = [emi_map.volume_physical_tags('neuron_%d' % i)['all'] for i in range(num_neurons)]

    mesh_ext = EmbeddedMesh(volume_marking_f, ext_Vtag)

    meshes_int = [EmbeddedMesh(volume_marking_f, n_Vtag) for n_Vtag in n_Vtags]

    # And current/Lagrange multiplier on each neuron
    meshes_neuron = [EmbeddedMesh(facet_marking_f, emi_map.surface_physical_tags('neuron_%d' % i).values())
                      for i in range(num_neurons)]

    # Cary over surface markers to extracellular mesh
    ext_boundaries = transfer_markers(mesh_ext, facet_marking_f)

    # Then there is space for each potential
    Ve = FunctionSpace(mesh_ext, 'CG', 1)
    Vis = [FunctionSpace(mesh_int, 'CG', 1) for mesh_int in meshes_int]
    # And current
    Qis = [FunctionSpace(mesh_neuron, 'CG', 1) for mesh_neuron in meshes_neuron]

    # The total space is
    W = [Ve] + Vis + Qis
    print 'W dim is', sum(Wi.dim() for Wi in W)

    # Build the block operator
    ue, uis, pis = TrialFunction(Ve), map(TrialFunction, Vis), map(TrialFunction, Qis)
    ve, vis, qis = TestFunction(Ve), map(TestFunction, Vis), map(TestFunction, Qis)
    # We will need trace operators
    # Rectricting ext to each neuron
    Tues = [Trace(ue, neuron) for neuron in meshes_neuron]
    Tves = [Trace(ve, neuron) for neuron in meshes_neuron]
    # Restricting each ui to its surface
    Tuis = [Trace(ui, neuron) for ui, neuron in zip(uis, meshes_neuron)]
    Tvis = [Trace(vi, neuron) for vi, neuron in zip(vis, meshes_neuron)]
    
    # And with each surface we need to associate a measure
    dx_ = [Measure('dx', domain=neuron) for neuron in meshes_neuron]
    # For setting up Nuemann bcs on extrac. potential we need
    ds_ = Measure('ds', domain=mesh_ext, subdomain_data=ext_boundaries)
    
    # Now the block operators
    a = block_form(W, 2)
    # Laplacians - external
    a[0][0] = Constant(ext_parameters['cond_ext'])*inner(grad(ue), grad(ve))*dx
    # Internal
    for i, (ui, vi) in enumerate(zip(uis, vis), 1):
        cond = neurons_parameters[i-1]['cond_int']
        a[i][i] = Constant(cond)*inner(grad(ui), grad(vi))*dx
    # Coupling
    for i, (Tve_i, Tue_i, p_i, q_i, dx_i) in enumerate(zip(Tves, Tues, pis, qis, dx_), 1 + num_neurons):
        a[0][i] = -inner(Tve_i, p_i)*dx_i
        # Do the symmetric block while here
        a[i][0] = -inner(Tue_i, q_i)*dx_i

    for (i, (pi, qi, dx_i)), (j, (Tui_j, Tvi_j)) in zip(enumerate(zip(pis, qis, dx_), 1+num_neurons),
                                                        enumerate(zip(Tuis, Tvis), 1)):
        a[j][i] = inner(Tvi_j, p_i)*dx_i
        a[i][j] = inner(Tui_j, q_i)*dx_i

    dt_fem = Constant(solver_parameters['dt_fem'])

    # Time step term
    for i, (p_i, q_i, n_props) in enumerate(zip(qis, pis, neurons_parameters), 1 + num_neurons):
        Cm = n_props['Cm']
        a[i][i] = -Constant(dt_fem/Cm)*inner(p_i, q_i)*dx 

    # Boundary conditions: grounded surfaces are Dirichlet
    #                      neumann surfaces are part of weak form and
    #                       - insulation means the integral is zero so
    #                         we only care about stimulated    
    grounded_tags = set(emi_map.surface_physical_tags('box').values())  # All
    grounded_tags.difference_update(set(emi_map.surface_physical_tags('box')[name]
                                        for name in ext_parameters['insulated_bcs']))
    
    # Bcs to be put explicitely on system
    Ve_bcs = [DirichletBC(Ve, Constant(0), ext_boundaries, tag)
              for tag in grounded_tags]
    # No other subspace has bcs; Vis, Qis
    W_bcs = [Ve_bcs] + [list() for _ in range(2*num_neurons)]

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
    # potential on the i-th neuron; They all s
    p0is = [interpolate(Constant(v_rest), Qi) for Qi in Qis]

    L = block_form(W, 1)
    L[0] = inner(Constant(0), ve)*dx + sum(inner(current, ve)*ds_(tag) for tag, current in stimulated_map.items())
    for i, (p0i, qi) in enumerate(zip(p0is, qis), 1+num_neurons):
        L[i] = inner(p0i, qi)*dx
    
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
        map_ = emi_map.surface_physical_tags('neuron_%d' % i)

        soma = tuple(map_[k] for k in map_ if 'soma' in k)
        dendrite = tuple(map_[k] for k in map_ if 'dend' in k)
        axon = tuple(map_[k] for k in map_ if 'axon' in k)

        ode_solver = MembraneODESolver(neuron.marking_function,
                                       soma=soma, axon=axon, dendrite=dendrite,
                                       problem_parameters=neurons_parameters[i],
                                       scale_factor=scale_factor,
                                       # In mortar we want ODE in vertices
                                       solver_parameters={'V_polynomial_family': 'Lagrange',
                                                          'V_polynomial_degree': 1,
                                                          'S_polynomial_family': 'Lagrange',
                                                          'S_polynomial_degree': 1})

        sim_duration = solver_parameters['sim_duration']
        assert sim_duration > 0.0
        interval = (0.0, sim_duration)

        # NOTE: a generator; nothing is computed so far
        ode_solutions = ode_solver.solve(interval, dt_ode)  # Potentials only
        neuron_solutions.append(ode_solutions)

    get_transmembrane_potentials = []
    for i, (Vi, Qi) in enumerate(zip(Vis, Qis)):
        
        ext_mat = PETScMatrix(trace_mat_no_restrict(Ve, Qi))
        int_mat = PETScMatrix(trace_mat_no_restrict(Vi, Qi))

        get_transmembrane_potentials.append(
            lambda ue, ui, Te=ext_mat, Ti=int_mat: -Te*ue.vector() + Ti*ui.vector()
        )

    # Solution in Ve x Vi x Q
    wh = ii_Function(W)
    # Set initial intracellular; we don't know about currents
    for i in range(1, 1+num_neurons):
        wh[i].vector().zero()
        wh[i].vector()[:] += v_rest  # Set resting potential

    # To have consistent API we wan to represent
    # i) the potential by one global P0 function on mesh
    # ii) transmembrane potential on the union of all neurons
    # iii) membrane curent as one function on the union of all neurons
    all_neurons_mesh = UnionMesh(meshes_neuron, no_overlap=True)
    current_out = UnionFunction(meshes_neuron, wh[(1+num_neurons):len(W)], all_neurons_mesh)
    v_out = UnionFunction(meshes_neuron, p0is, all_neurons_mesh)

    # NOTE: Get the potentials as P0
    u_outs = [Function(FunctionSpace(m, 'DG', 0)) for m in [mesh_ext] + meshes_int]
    # Fill the uouts
    for i, u_outi in enumerate(u_outs):
        Xi = u_outi.function_space()
        # Here the P0 is set as an L2 projection of P1 to P0
        u_outi.vector()[:] = assemble((1/CellVolume(Xi.mesh()))*inner(wh[i], TestFunction(Xi))*dx)
    # And then u_out is on global (mesh)
    u_out = UnionFunction([mesh_ext] + meshes_int, u_outs, mesh)    
    
    # To get initial state
    yield 0, u_out, current_out

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

            # Update global
            v_out.sync()
            current_out.sync()
            # Fill the uouts
            for i, u_outi in enumerate(u_outs):
                Xi = u_outi.function_space()
                # L2 locally
                u_outi.vector()[:] = assemble((1/CellVolume(Xi.mesh()))*inner(wh[i], TestFunction(Xi))*dx)
            # Global sync
            u_out.sync()
            
            e = lambda x: sqrt(abs(assemble(inner(x, x)*dx)))
            print e(u_out), e(v_out), e(current_out), '<<<<<'
                
            yield t1, u_out, current_out

            # Get transmembrane potential to ODE for next round
            for i in range(num_neurons):
                odes[i][1].vector().zero()
                odes[i][1].vector().axpy(1, p0is[i].vector())
