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


def solve_poisson(mesh_path, problem_parameters, solver_parameters):
    '''Poisson solver in Hdiv formulation - just with the stimulus on probe site'''
    mesh, facet_marking_f, volume_marking_f, tags = load_mesh(mesh_path)

    cell = mesh.ufl_cell()
    # We have 3 spaces S for sigma = -kappa*grad(u)   [~electric field]
    #                  U for potential u
    #                  Q for transmebrane potential p
    Sel = FiniteElement('RT', cell, 1)
    Vel = FiniteElement('DG', cell, 0)

    W = FunctionSpace(mesh, MixedElement([Sel, Vel]))
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    soma, axon, dendrite = {1}, tags['axon'], tags['dendrite']
    assert not tags['axon']
    assert not tags['dendrite']

    insulated_surfaces = tags['probe_surfaces']
    grounded_surfaces = {5, 6}
    
    # We now want to add all but the stimulated probe site to insulated
    # surfaces and for the stimulated probe prescribe ...
    stimulated_site = problem_parameters['stimulated_site']
    all_sites = tags['contact_surfaces']
    assert stimulated_site in all_sites

    all_sites.remove(stimulated_site)
    insulated_surfaces.update(all_sites)  # Rest is insulated

    bc_insulated = [DirichletBC(W.sub(0), Constant((0, 0, 0)), facet_marking_f, tag)
                    for tag in insulated_surfaces]
    # NOTE: (0, 0, 0) means that the dof is set based on (0, 0, 0).n
    # Add the stimulated site
    site_current = problem_parameters['site_current']
    assert len(site_current) == 3
    bc_insulated.append(DirichletBC(W.sub(0), site_current, facet_marking_f, stimulated_site))

    # Integrate over entire volume
    dx = Measure('dx')

    cond_int = Constant(problem_parameters['cond_int'])
    cond_ext = Constant(problem_parameters['cond_ext'])

    a = ((1/cond_ext)*inner(sigma, tau)*dx()+
         - inner(div(tau), u)*dx() 
         - inner(div(sigma), v)*dx())

    L = inner(Constant(0), v)*dx()

    # Get the linear system
    assembler = SystemAssembler(a, L, bcs=bc_insulated)
    A, b = Matrix(), Vector()
    assembler.assemble(A) 
    assembler.assemble(b)

    wh = Function(W)
    solve(A, wh.vector(), b)

    return wh.split(deepcopy=True)
