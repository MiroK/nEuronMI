from linear_algebra import LinearSystemSolver
from transferring import SubMeshTransfer
from embedding import EmbeddedMesh
from membrane import ODESolver
from aux import load_mesh
from dolfin import *
import operator


# Optimizations
parameters['form_compiler']['representation'] = 'uflacs'
parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['cpp_optimize_flags'] = '-O3 -ffast-math -march=native'
parameters['ghost_mode'] = 'shared_facet'


def neuron_solver(mesh_path, emi_map, problem_parameters, solver_parameters):
    '''
    Solver for the Hdiv formulation of the EMI equations
    
    mesh_path: str that is the path to HDF5File containing mesh, ...
    emi_map: EMIEntityMap of the mesh
    problem_parameters: dict specifying the following

      For each neuron (neuron_i) the I_ion, cond[uctivity], C_m, parameters for stim[_*]ulation params
      For exterior (external) cond[uctivity], names of insulated exterior boundaries
      For probe stimulated_sites (named) and their currents

    solver_parameters: time_step, dt (of EMI), dt_ode
    '''    
    mesh, volume_marking_f, facet_marking_f = load_mesh(mesh_path)

    num_neurons = emi_map.num_neurons
    # Do we have properties for each one
    neuron_props = [problem_parameters['neuron_%d'] % i for i in range(num_neurons)]
    ext_props = problem_parameters['external']

    cell = mesh.ufl_cell()
    # We have 3 spaces S for sigma = -kappa*grad(u)   [~electric field]
    #                  U for potential u
    #                  Q for transmebrane potential p;
    Sel = FiniteElement('RT', cell, 1)
    Vel = FiniteElement('DG', cell, 0)
    Qel = FiniteElement('Discontinuous Lagrange Trace', cell, 0)

    W = FunctionSpace(mesh, MixedElement([Sel, Vel, Qel]))
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
    p0 = Function(Q)              # Previous transm potential now 0

    # The weak form
    # kappa**-1 * (sigma, tau)*dx - (div tau, u)*dx + (tau.n, p)*dS = 0
    # -(div sigma, v)*dx                                            = 0
    # (sigma.n - Cm/dt*p, q)*dS                                     = (I_ion - Cm/dt*p0)*dS

    # Extract volumes tags for volume and neurons
    ext_Vtag = emi_map.volume_physical_tags('external')['all']
    n_Vtags = [emi_map.volume_physical_tags('neuron_%d' % i)['all'] for i in range(num_neurons)]
    
    a = ((1/Constant(ext_props['cond']))*inner(sigma, tau)*dx(ext_Vtag)
         - inner(div(tau), u)*dx(ext_Vtag)
         - inner(div(sigma), v)*dx(ext_Vtag))
    # Add neurons
    for n_Vtag, n_props in zip(n_Vtags, neuron_props):
        a += ((1/Constant(n_props['cond']))*inner(sigma, tau)*dx(n_Vtag)+
              - inner(div(tau), u)*dx(n_Vtag)
              - inner(div(sigma), v)*dx(n_Vtag))

    dt_fem = solver_parameters['dt_fem']
    # Extract surface tags for surface contribs of the neurons.
    # NOTE: here the distanction between surfaces does of neuron does
    # not matter
    n_Stags = [emi_map.surface_physical_tags('neuron_%d' % i).keys() for i in range(num_neurons)]

    for n_Stag, n_props in zip(n_Stags, neuron_props):
        a += sum(inner(p('+'), dot(tau('+'), n))*dS(i) for i in n_Stag)
        a += sum(inner(q('+'), dot(sigma('+'), n))*dS(i) for i in n_Stag)
        a += -sum(Constant(n_props['C_m']/dt_fem)*inner(q('+'), p('+'))*dS(i) for i in n_Stag)

    # Rhs contributions
    L = 0
    for n_Stag, n_props in zip(n_Stags, neuron_props):
        L += sum(inner(q('+'), n_props['I_ion']-Constant(n_props['C_m']/dt_fem)*p0('+'))*dS(i)
                 for i in neuron_surfaces)

    # Boundary conditions: grounded surfaces are neumann and we don't do
    # anything special there. Insulated sites and the stimated site(s) of
    # the probe are Dirichlet. Additional Dirichlet bcs contrain DLT dofs
    insulated_tags = [emi_map.surface_physical_tags('external')[name] for name in ext_props['bcs']]
    # NOTE: (0, 0, 0) means that the dof is set based on (0, 0, 0).n
    bc_insulated = [DirichletBC(W.sub(0), Constant((0, 0, 0)), facet_marking_f, tag)
                    for tag in insulated_surfaces]

    # Add the stimulated site
    if 'probe' in emi_map:
        probe_params = problem_parameters['probe']
        
        site_currents = probe_params['site_currents']
        stim_sites = [emi_map.physical_surfaces_tags('probe')[name]
                      for name in probe_params['stimulated_siters']]

        bc_stimulated = [DirichletBC(W.sub(0), site, facet_marking_f, current)
                         for site, current in zip(stim_sites, site_currents)]
        # From the system they are the same
        bc_insulated.extend(bc_stimulated)

    not_neuron_surfaces = set(facet_marking_f.array()) - set(sum(n_Stags, []))
    # A specific of the setup is that the facet space is too large. It
    # should be defined only on the neuron surfaces but it is defined
    # everywhere instead. So the not neuron part should be set to 0
    bc_constrained = [DirichletBC(W.sub(2), Constant(0), facet_marking_f, tag) for tag in not_neuron_surfaces]

    assembler = SystemAssembler(a, L, bcs=bc_insulated+bc_constrained)
    A, b = Matrix(), Vector()
    assembler.assemble(A) 
    assembler.assemble(b)
    # import numpy as np
    # print np.min(np.abs(np.linalg.eigvalsh(A.array())))
    la_solver = LinearSystemSolver(A, W, solver_parameters)
    
    dt_ode = solver_parameters['dt_ode']
    assert dt_ode <= dt_fem(0)
    # Setup neuron
    fem_ode_sync = int(dt_fem(0)/dt_ode)

    # For each neuron we wa

    for i, neuron_surfaces in enumerate(n_Stags):
        # Ode solver. Defined on the neuron mesh
        neuron_surf_mesh, neuron_subdomains = EmbeddedMesh(facet_marking_f, neuron_surfaces)

        # FIXME: union axon_*, dendrite=*
        soma = 1
        dendrite = 1
        axon = 1
        
        ode_solver = ODESolver(neuron_subdomains,
                               soma=soma, axon=axons, dendrite=dendrite,
                               problem_parameters=problem_parameters['neuron_%d' % i])

        Tstop = problem_parameters['Tstop']; assert Tstop > 0.0
        interval = (0.0, Tstop)
    
        # NOTE: a generator; nothing is computed so far
        ode_solutions = ode_solver.solve(interval, dt_ode)  # Potentials only

        transfer = SubMeshTransfer(mesh, neuron_surf_mesh)
        # The ODE solver talks to the worlk via chain: Q_neuron <-> Q <- W
        Q_neuron = ode_solver.V
        p0_neuron = Function(Q_neuron)

        # Between DLT mesh and submesh space
        assign_toQ_neuron_fromQ = transfer.compute_map(Q_neuron, Q, strict=False)
        assign_toQ_fromQ_neuron = transfer.compute_map(Q, Q_neuron, strict=False)

    # Get the linear system
    # From component to DLT on mesh
    toQ_fromW2 = FunctionAssigner(Q, W.sub(2))


    w = Function(W)
    # Finally for postprocessing we return the current time, potential
    # and membrane current
    V = FunctionSpace(mesh, Vel)
    u_out = Function(V)
    toV_fromW1 = FunctionAssigner(V, W.sub(1))
    toV_fromW1.assign(u_out, w.sub(1))

    # One value per cell of the neuron surface mesh
    current_out, current_aux = map(Function, (Q_neuron, Q))
    w_aux = Function(W)
    current_form = sum(1./FacetArea(mesh)('+')*inner(dot(w.sub(0)('+'), n), q('+'))*dS(i)
                       for i in neuron_surfaces)
    current_form += inner(Constant(0), v)*dx(neuron_int)  # Fancy zero for orientation
    # The idea here is that assembling the current form gives the right
    # dof values to assign to the DLT space (evals at cell midpoints).
    # Then we reduce as normal to the subcomponent and submesh space
    w_aux.vector()[:] = assemble(current_form)
    toQ_fromW2.assign(current_aux, w_aux.sub(2))
    assign_toQ_neuron_fromQ(current_out, current_aux)
    
    # To get initial state
    yield 0, u_out, current_out, A.size(0)
                    
    step_count = 0
    for ((t0, t1), ode_solution) in ode_solutions:
        step_count += 1
        info('Time is (%g, %g)' % (t0, t1))
        if step_count == fem_ode_sync:
            step_count = 0
            # ODE -> p0_neuron
            p0_neuron.assign(ode_solution)
            # Upscale p0_neuron->p0
            assign_toQ_fromQ_neuron(p0, p0_neuron)
        
            # We could have changing in time simulation
            if 't' in site_current.user_parameters:
                site_current.t = float(t1)
            # Assemble right-hand side (changes with time, so need to reassemble)                
            assembler.assemble(b)  # Also applies bcs
      
            # New (sigma, u, p) ...
            info('\tSolving linear system of size %d' % A.size(0))
            info('\tNumber of true unknowns %d' % (A.size(0) - ncstr_dofs))
            la_solver.solve(w.vector(), b)

            # Update u_out and current_out for output
            toV_fromW1.assign(u_out, w.sub(1))
            # NOTE: the current_form is points to w which has been updated by solve
            w_aux.vector()[:] = assemble(current_form)
            toQ_fromW2.assign(current_aux, w_aux.sub(2))
            assign_toQ_neuron_fromQ(current_out, current_aux)

            yield t1, u_out, current_out, A.size(0)
            
            # Now transfer the new transm potential down to ode ...
            toQ_fromW2.assign(p0, w.sub(2))         # Compt to Q
            assign_toQ_neuron_fromQ(p0_neuron, p0)  # To membrane space
            ode_solution.assign(p0_neuron)         # As IC for ODE
