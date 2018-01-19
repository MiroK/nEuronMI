from linear_algebra import LinearSystemSolver
from transferring import SubMeshTransfer
from embedding import EmbeddedMesh
from membrane import ODESolver
from aux import load_mesh
from dolfin import *


# Optimizations
parameters['form_compiler']['representation'] = 'uflacs'
parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['cpp_optimize_flags'] = '-O3 -ffast-math -march=native'
parameters['ghost_mode'] = 'shared_facet'


def neuron_solver(mesh_path, problem_parameters, solver_parameters):
    '''Solver for the Hdiv formulation of the EMI equations'''
    mesh, facet_marking_f, volume_marking_f, tags = load_mesh(mesh_path)

    cell = mesh.ufl_cell()
    # We have 3 spaces S for sigma = -kappa*grad(u)   [~electric field]
    #                  U for potential u
    #                  Q for transmebrane potential p
    Sel = FiniteElement('RT', cell, 1)
    Vel = FiniteElement('DG', cell, 0)
    Qel = FiniteElement('Discontinuous Lagrange Trace', cell, 0)

    W = FunctionSpace(mesh, MixedElement([Sel, Vel, Qel]))
    sigma, u, p = TrialFunctions(W)
    tau, v, q = TestFunctions(W)

    soma, axon, dendrite = {1}, tags['axon'], tags['dendrite']
    insulated_surfaces = tags['probe_surfaces']
    
    grounded_surfaces = {5, 6}
    neuron_surfaces = soma | axon | dendrite
    other_surfaces = {0}
    # Boundary conditions are that (for now) (4, 41) are insulated, i.e
    # sigma*grad(u).n = 0, and (5, 6) are grounded meaning u = 0. In this
    # formulation the grounding condition is enforced weakly meaning that
    # integrals over 5, 6 are not included in the weak form (vanish). On
    # the other hand insulated condition is enforced strongly, that is
    # on the function space level
    bc_insulated = [DirichletBC(W.sub(0), Constant((0, 0, 0)), facet_marking_f, tag)
                    for tag in insulated_surfaces]
    # NOTE: (0, 0, 0) means that the dof is set based on (0, 0, 0).n

    all_surfaces = insulated_surfaces | grounded_surfaces | neuron_surfaces | other_surfaces
    # A specific of the setup is that the facet space is too large. It
    # should be defined only on the neuron surfaces but it is defined
    # everywhere instead. So the not neuron part should be set to 0
    bc_constrained = [DirichletBC(W.sub(2), Constant(0), facet_marking_f, tag)
                      for tag in (all_surfaces - neuron_surfaces)]

    # To integrate over inside and outside of the neuron we define a volume
    # measure. Note that interior is mared as 1 and outside is 2
    neuron_int = 1
    neuron_ext = 2
    dx = Measure('dx', domain=mesh, subdomain_data=volume_marking_f)

    # We will also need a measure for integratin over the neuron surface
    dS = Measure('dS', domain=mesh, subdomain_data=facet_marking_f)
    
    # And finally a normal fo the INTERIOR surface. Note that 1, 2 marking
    # of volume makes 2 cells the '+' cells w.r.t to neuron survace. n('+')
    # would therefore be their outer normal (that is an outer normal of the
    # outside). ('-') makes the orientation right
    n = FacetNormal(mesh)('-')

    # Now onto the weak form
    # Load up user parameters of the problem and the solver
    C_m = Constant(problem_parameters['C_m'])
    cond_int = Constant(problem_parameters['cond_int'])
    cond_ext = Constant(problem_parameters['cond_ext'])
    # FIXME: is ionic current always constant in this application
    I_ion = Constant(problem_parameters['I_ion'])    
    dt_fem = Constant(solver_parameters['dt_fem'])

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

    a = ((1/cond_int)*inner(sigma, tau)*dx(neuron_int)+(1/cond_ext)*inner(sigma, tau)*dx(neuron_ext)
         - inner(div(tau), u)*dx(neuron_int) - inner(div(tau), u)*dx(neuron_ext)
         + sum(inner(p('+'), dot(tau('+'), n))*dS(i) for i in neuron_surfaces)
         - inner(div(sigma), v)*dx(neuron_int) - inner(div(sigma), v)*dx(neuron_ext)
         + sum(inner(q('+'), dot(sigma('+'), n))*dS(i) for i in neuron_surfaces)
         - sum((C_m/dt_fem)*inner(q('+'), p('+'))*dS(i) for i in neuron_surfaces))

    L = sum(inner(q('+'), I_ion-(C_m/dt_fem)*p0('+'))*dS(i) for i in neuron_surfaces)

    # Ode solver. Defined on the neuron mesh
    neuron_surf_mesh, neuron_subdomains = EmbeddedMesh(facet_marking_f, neuron_surfaces)
    # For simplicity there shall be the same physics on hillock as on the
    # connected dendrite/exon. So we rename 31->3 and 21->2
    if 31 in dendrite:
        dendrite.remove(31)
        for c in SubsetIterator(neuron_sobdomains, 31):
            neuron_soldomains[c] = 3

    if 21 in axon:
        axon.remove(21)
        for c in SubsetIterator(neuron_sobdomains, 21):
            neuron_soldomains[c] = 2

    dt_ode = solver_parameters['dt_ode']
    assert dt_ode < dt_fem(0)
    
    # FIXME(Karoline): What are the appropriate models, excitation,
    #                  and other parameters? Backward Euler etc
    #                  Feel free to change the API
    ode_solver, neuron_model = ODESolver(neuron_subdomains,
                                         soma=next(iter(soma)),
                                         axon=next(iter(axon)),
                                         dendrite=next(iter(dendrite)),
                                         parameters=solver_parameters)

    transfer = SubMeshTransfer(mesh, neuron_surf_mesh)
    # The ODE solver talks to the worlk via chain: VS <-> Q_neuron <-> Q <- W
    ODE_space = ode_solver.VS  # Here the first is the transm. potental
    Q_neuron = ODE_space.sub(0).collapse()    
    p0_neuron = Function(Q_neuron)

    # From component to DLT on mesh
    toQ_fromW2 = FunctionAssigner(Q, W.sub(2))
    # Between DLT mesh and submesh space
    assign_toQ_neuron_fromQ = transfer.compute_map(Q_neuron, Q, strict=False)
    assign_toQ_fromQ_neuron = transfer.compute_map(Q, Q_neuron, strict=False)
    # Between submesh space and internals of the ODE solver
    toODE_fromQ_neuron = None  # FIXME: for now they are defined only
    toQ_neuron_fromODE = None  # declared here and defined later

    # And finally onto the solution loop
    # ODE
    Tstop = problem_parameters['Tstop']; assert Tstop > 0.0
    interval = (0.0, Tstop)
    
    fem_ode_sync = int(dt_fem(0)/dt_ode)
    # NOTE: a generator; nothing is computed so far
    ode_solutions = ode_solver.solve(interval, dt_ode)
    # Set up initial conditions
    (ode_solution0, _) = ode_solver.solution_fields()
    ode_solution0.assign(neuron_model.initial_conditions())

    # Get the linear system
    assembler = SystemAssembler(a, L, bcs=bc_insulated+bc_constrained)
    A, b = Matrix(), Vector()
    assembler.assemble(A)
    assembler.assemble(b)

    # And its solver
    la_solver = LinearSystemSolver(A, W, solver_parameters)

    w = Function(W)
    # Finally for postprocessing we return the potential and current time
    V = FunctionSpace(mesh, Vel)
    u_out = Function(V)
    toV_fromW1 = FunctionAssigner(V, W.sub(1))
    toV_fromW1.assign(u_out, w.sub(1))
    # To get initial state
    yield 0, u_out
                    
    step_count = 0
    for ((t0, t1), ode_solution) in ode_solutions:
        step_count += 1
        info('Time is (%g, %g)' % (t0, t1))
        if step_count == fem_ode_sync:
            step_count = 0
            # ODE(0) -> p0_neuron
            if toQ_neuron_fromODE is None:
                toQ_neuron_fromODE = FunctionAssigner(p0_neuron.function_space(),
                                                      ode_solution.function_space().sub(0))
            toQ_neuron_fromODE.assign(p0_neuron, ode_solution.sub(0))
            # Upscale p0_neuron->p0
            assign_toQ_fromQ_neuron(p0, p0_neuron)
        
            # Assemble right-hand side (changes with time, so need to reassemble)
            assembler.assemble(b)  # Also applies bcs
            # New (sigma, u, p) ...
            info('\tSolving linear system of size %d' % A.size(0))
            la_solver.solve(w.vector(), b)

            # Update u_out for output. FIXME: is t1 the right time here?
            toV_fromW1.assign(u_out, w.sub(1))
            yield t1, u_out
            
            # Now transfer the new transm potential down to ode ...
            toQ_fromW2.assign(p0, w.sub(2))         # Compt to Q
            assign_toQ_neuron_fromQ(p0_neuron, p0)  # To membrane space
            if toODE_fromQ_neuron is None:
                toODE_fromQ_neuron = FunctionAssigner(ode_solution.function_space().sub(0),
                                                      p0_neuron.function_space())
            toODE_fromQ_neuron.assign(ode_solution.sub(0), p0_neuron)  # As IC for ODE
