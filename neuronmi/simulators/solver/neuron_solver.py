from neuronmi.simulators.solver.aux import SiteCurrent, surface_normal
from neuronmi.simulators.solver.linear_algebra import LinearSystemSolver
from neuronmi.simulators.solver.transferring import SubMeshTransfer
from neuronmi.simulators.solver.embedding import EmbeddedMesh
from neuronmi.simulators.solver.membrane import MembraneODESolver
from neuronmi.mesh.mesh_utils import load_h5_mesh
import numpy as np
import itertools

from dolfin import *

# Optimizations
parameters['form_compiler']['representation'] = 'uflacs'
parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['cpp_optimize_flags'] = '-O3 -ffast-math -march=native'
parameters['ghost_mode'] = 'shared_facet'


def neuron_solver(mesh_path, emi_map, problem_parameters, scale_factor=None, verbose=False):
    '''
    Solver for the Hdiv formulation of the EMI equations
    
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

    # neuron_props = [problem_parameters['neuron_%d' % i] for i in range(num_neurons)]
    # ext_props = problem_parameters['external']

    cell = mesh.ufl_cell()
    # We have 3 spaces S for sigma = -kappa*grad(u)   [~electric field]
    #                  U for potential u
    #                  Q for transmebrane potential p;
    Sel = FiniteElement('RT', cell, 1)
    Vel = FiniteElement('DG', cell, 0)
    Qel = FiniteElement('Discontinuous Lagrange Trace', cell, 0)

    W = FunctionSpace(mesh, MixedElement([Sel, Vel, Qel]))
    print('PDE part will be solved for %d unknowns' % W.dim())
    sigma, u, p = TrialFunctions(W)
    tau, v, q = TestFunctions(W)

    # To integrate over inside and outside of the neurons we define a volume
    dx = Measure('dx', domain=mesh, subdomain_data=volume_marking_f)
    # We will also need a measure for integratin over the neuron surfaces
    dS = Measure('dS', domain=mesh, subdomain_data=facet_marking_f)
    # Orient normal so that it is outer normal of neurons
    n = FacetNormal(mesh)('+')

    # Everything is driven by membrane response. This will be updated
    # by the ode solver. The ode solver will work on proper space defined
    # only on the neuron. The solution shall then be taked to a facet
    # space Q (think 3rd component of W). Finally W mu
    Q = FunctionSpace(mesh, Qel)  # Everywhere
    p0 = Function(Q)  # Previous transm potential now 0

    # The weak form
    # kappa**-1 * (sigma, tau)*dx - (div tau, u)*dx + (tau.n, p)*dS = 0
    # -(div sigma, v)*dx                                            = 0
    # (sigma.n - Cm/dt*p, q)*dS                                     = (I_ion - Cm/dt*p0)*dS

    # Extract volumes tags for volume and neurons
    ext_Vtag = emi_map.volume_physical_tags('external')['all']
    n_Vtags = [emi_map.volume_physical_tags('neuron_%d' % i)['all'] for i in range(num_neurons)]

    a = ((1 / Constant(ext_parameters['cond_ext'])) * inner(sigma, tau) * dx(ext_Vtag)
         - inner(div(tau), u) * dx(ext_Vtag)
         - inner(div(sigma), v) * dx(ext_Vtag))
    # Add neurons
    for n_Vtag, n_props in zip(n_Vtags, neurons_parameters):
        a += ((1 / Constant(n_props['cond_int'])) * inner(sigma, tau) * dx(n_Vtag) +
              - inner(div(tau), u) * dx(n_Vtag)
              - inner(div(sigma), v) * dx(n_Vtag))

    dt_fem = Constant(solver_parameters['dt_fem'])
    # Extract surface tags for surface contribs of the neurons.
    # NOTE: here the distanction between surfaces does of neuron does
    # not matter
    n_Stags = map(list,
                  [emi_map.surface_physical_tags('neuron_%d' % i).values() for i in range(num_neurons)])
    n_Stags = list(n_Stags)

    for n_Stag, n_props in zip(n_Stags, neurons_parameters):
        a += sum(inner(p('+'), dot(tau('+'), n)) * dS(i) for i in n_Stag)
        a += sum(inner(q('+'), dot(sigma('+'), n)) * dS(i) for i in n_Stag)
        a += -sum(Constant(n_props['Cm'] / dt_fem) * inner(q('+'), p('+')) * dS(i) for i in n_Stag)

    iterator = iter(zip(n_Stags, neurons_parameters))
    # Rhs contributions
    n_Stag, n_props = next(iterator)
    L = sum(inner(q('+'), I_ion - Constant(n_props['Cm'] / dt_fem) * p0('+')) * dS(i)
            for i in n_Stag)

    for n_Stag, n_props in iterator:
        L += sum(inner(q('+'), I_ion - Constant(n_props['Cm'] / dt_fem) * p0('+')) * dS(i)
                 for i in n_Stag)

    # Boundary conditions: grounded surfaces are neumann and we don't do
    # anything special there. Insulated sites and the stimated site(s) of
    # the probe are Dirichlet. Additional Dirichlet bcs contrain DLT dofs
    insulated_tags = [emi_map.surface_physical_tags('box')[name] for name in ext_parameters['insulated_bcs']]
    # NOTE: (0, 0, 0) means that the dof is set based on (0, 0, 0).n
    bc_insulated = [DirichletBC(W.sub(0), Constant((0, 0, 0)), facet_marking_f, tag)
                    for tag in insulated_tags]

    # The site current are normal*magnitude where the normal is pointing
    # into the inside of the probe. That is, wrt box that contains it it
    # is outer normal.
    inside_point = mesh.coordinates().min(axis=0)  # In the exterior of probe

    site_currents = []
    # Add the stimulated site
    if 'probe' in emi_map.surfaces:
        probe_surfaces = emi_map.surface_physical_tags('probe')  # Dict
        stim_sites = []  # names
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

                    for name in probe_parameters['stimulated_sites']:
                        tag = probe_surfaces[name]
                        stim_sites.append(tag)
                    # Construct normal*I expressions for every site
                    site_currents = [SiteCurrent(I=current, n=surface_normal(site, facet_marking_f, inside_point),
                                                 degree=1)
                                     for site, current in zip(stim_sites, site_currents)]
                    # Now the bcs
                    bc_stimulated = [DirichletBC(W.sub(0), current, facet_marking_f, site)
                                     for site, current in zip(stim_sites, site_currents)]
                    # From the system they are the same
                    bc_insulated.extend(bc_stimulated)

        # Sites of the probe that are not stimulated are insulated
        insulated_probe_sites = set(probe_surfaces.values()) - set(stim_sites)

        # Enforce on PDE
        bc_insulated.extend(DirichletBC(W.sub(0), Constant((0, 0, 0)), facet_marking_f, site)
                            for site in insulated_probe_sites)

    all_neuron_surfaces = set(sum(n_Stags, []))
    not_neuron_surfaces = set(facet_marking_f.array()) - all_neuron_surfaces
    # A specific of the setup is that the facet space is too large. It
    # should be defined only on the neuron surfaces but it is defined
    # everywhere instead. So the not neuron part should be set to 0
    bc_constrained = [DirichletBC(W.sub(2), Constant(0), facet_marking_f, tag) for tag in not_neuron_surfaces]

    assembler = SystemAssembler(a, L, bcs=bc_insulated + bc_constrained)
    A, b = Matrix(), Vector()
    assembler.assemble(A)
    assembler.assemble(b)

    # Not singular
    # import numpy as np
    # print np.min(np.abs(np.linalg.eigvalsh(A.array())))
    la_solver = LinearSystemSolver(A, W, solver_parameters)

    dt_ode = solver_parameters['dt_ode']
    assert dt_ode <= dt_fem(0)
    # Setup neuron
    fem_ode_sync = int(dt_fem(0) / dt_ode)

    # Mesh of all neurons; individual are its submesh
    neuron_surf_mesh = EmbeddedMesh(facet_marking_f, list(all_neuron_surfaces))
    neurons_subdomains = neuron_surf_mesh.marking_function
    # It is on this mesh that ode will update transmemebrane current and
    # talk with pde
    Q_neuron = FunctionSpace(neuron_surf_mesh, 'DG', 0)  # P0 on surface <-> DLT on facets

    transfer = SubMeshTransfer(mesh, neuron_surf_mesh)
    # The ODE solver talks to the worlk via chain: Q_neuron <-> Q <- W
    p0_neuron = Function(Q_neuron)
    # Between DLT mesh and submesh space
    assign_toQ_neuron_fromQ = transfer.compute_map(Q_neuron, Q, strict=False)
    assign_toQ_fromQ_neuron = transfer.compute_map(Q, Q_neuron, strict=False)
    # From component to DLT on mesh
    toQ_fromW2 = FunctionAssigner(Q, W.sub(2))

    toQin_fromQns, toQn_fromQins, p0is = [], [], []
    # p0i \in Qi <-> Q_neuron \ni p0_neuron
    neuron_solutions = []
    for i, neuron_surfaces in enumerate(n_Stags):
        # Pick the nueuron from neuron collection
        ni_mesh = EmbeddedMesh(neurons_subdomains, neuron_surfaces)
        ni_subdomains = ni_mesh.marking_function

        map_ = emi_map.surface_physical_tags('neuron_%d' % i)

        soma = tuple(map_[k] for k in map_ if 'soma' in k)
        dendrite = tuple(map_[k] for k in map_ if 'dend' in k)
        axon = tuple(map_[k] for k in map_ if 'axon' in k)

        ode_solver = MembraneODESolver(ni_subdomains,
                                       soma=soma, axon=axon, dendrite=dendrite,
                                       problem_parameters=neurons_parameters[i],
                                       scale_factor=scale_factor)

        sim_duration = solver_parameters['sim_duration']
        assert sim_duration > 0.0
        interval = (0.0, sim_duration)

        # NOTE: a generator; nothing is computed so far
        ode_solutions = ode_solver.solve(interval, dt_ode)  # Potentials only
        neuron_solutions.append(ode_solutions)

        transfer = SubMeshTransfer(neuron_surf_mesh, ni_mesh)
        # Communication between neuron and the collection
        Qi_neuron = ode_solver.V
        p0i_neuron = Function(Qi_neuron)

        # Between DLT mesh and submesh space
        assign_toQin_fromQn = transfer.compute_map(Qi_neuron, Q_neuron, strict=False)
        assign_toQn_fromQin = transfer.compute_map(Q_neuron, Qi_neuron, strict=False)

        toQin_fromQns.append(assign_toQin_fromQn)
        toQn_fromQins.append(assign_toQn_fromQin)
        p0is.append(p0i_neuron)

    V = FunctionSpace(mesh, Vel)
    # Finally for postprocessing we return the current time, potential
    # and membrane current    
    u_out = Function(V)
    u_out_values = u_out.vector().get_local()

    array = volume_marking_f.array()
    # Set potential inside neurons ...
    for n_Vtag in n_Vtags:
        # Rest if inside else the value that was there. So this is like or
        u_out_values[:] = np.where(array == n_Vtag, v_rest, u_out_values)
    u_out.vector().set_local(u_out_values)
    u_out.vector().apply('insert')

    w = Function(W)
    FunctionAssigner(W.sub(1), V).assign(w.sub(1), u_out)
    # Keep the assigner around as we'll go out in the loop
    toV_fromW1 = FunctionAssigner(V, W.sub(1))
    toV_fromW1.assign(u_out, w.sub(1))

    # One value per cell of the neuron surface mesh
    current_out, current_aux = map(Function, (Q_neuron, Q))
    w_aux = Function(W)
    current_form = sum(1. / FacetArea(mesh)('+') * inner(dot(w.sub(0)('+'), n), q('+')) * dS(i)
                       for i in all_neuron_surfaces)

    current_form += inner(Constant(0), v) * dx(ext_Vtag)  # Fancy zero for orientation
    # The idea here is that assembling the current form gives the right
    # dof values to assign to the DLT space (evals at cell midpoints).
    # Then we reduce as normal to the subcomponent and submesh space
    w_aux.vector()[:] = assemble(current_form)
    toQ_fromW2.assign(current_aux, w_aux.sub(2))
    assign_toQ_neuron_fromQ(current_out, current_aux)

    # To get initial state
    yield 0, u_out, current_out, p0_neuron

    neuron_solutions = itertools.izip(*neuron_solutions)

    step_count = 0
    for odes in neuron_solutions:
        step_count += 1

        (t0, t1) = odes[0][0]
        print('Time is (%g, %g)' % (t0, t1))
        if step_count == fem_ode_sync:
            step_count = 0
            # From individual neuron to collection
            for i in range(num_neurons):
                # FIXME: does this override?
                toQn_fromQins[i](p0_neuron, odes[i][1])
            # Upscale p0_neuron->p0
            assign_toQ_fromQ_neuron(p0, p0_neuron)

            # We could have changing in time simulation
            for I in site_currents:
                if 't' in I:
                    I.t = float(t1)
            # Assemble right-hand side (changes with time, so need to reassemble)                
            assembler.assemble(b)  # Also applies bcs
            # New (sigma, u, p) ...
            if verbose:
                print('\tSolving linear system of size %d' % A.size(0))
            la_solver.solve(w.vector(), b)

            # Update u_out and current_out for output
            toV_fromW1.assign(u_out, w.sub(1))
            # NOTE: the current_form is points to w which has been updated by solve
            w_aux.vector()[:] = assemble(current_form)
            toQ_fromW2.assign(current_aux, w_aux.sub(2))
            assign_toQ_neuron_fromQ(current_out, current_aux)

            # Now transfer the new transm potential down to ode ...
            toQ_fromW2.assign(p0, w.sub(2))  # Compt to Q
            assign_toQ_neuron_fromQ(p0_neuron, p0)  # To membrane space

            yield t1, u_out, current_out, p0_neuron

            for i in range(num_neurons):
                toQin_fromQns[i](p0is[i], p0_neuron)
                odes[i][1].assign(p0is[i])
