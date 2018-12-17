from linear_algebra import LinearSystemSolver
from transferring import SubMeshTransfer
from embedding import EmbeddedMesh
from membrane import ODESolver
from aux import load_mesh
import numpy as np
from dolfin import *
import operator


# Optimizations
parameters['form_compiler']['representation'] = 'uflacs'
parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['cpp_optimize_flags'] = '-O3 -ffast-math -march=native'
parameters['ghost_mode'] = 'shared_facet'


def solve_poisson_mixed(mesh_path, problem_parameters, solver_parameters):
    '''Poisson solver in Hdiv formulation - just with the stimulus on probe site'''
    mesh, facet_marking_f, volume_marking_f, tags = load_mesh(mesh_path)

    cell = mesh.ufl_cell()
    Sel = FiniteElement('RT', cell, 1)
    Vel = FiniteElement('DG', cell, 0)

    W = FunctionSpace(mesh, MixedElement([Sel, Vel]))
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    soma, axon, dendrite = {1}, tags['axon'], tags['dendrite']
    assert not tags['axon']
    assert not tags['dendrite']

    insulated_surfaces = tags['probe_surfaces']
    grounded_surfaces = {5, 6}  # Enforced weakly
    
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

    # FIXME: point sources

    wh = Function(W)
    solve(A, wh.vector(), b)

    return wh.split(deepcopy=True)


class PoissonSolver(object):
    '''Static Poisson solver with stimulated contact site and point sources'''
    def __init__(self, mesh_path, problem_parameters, solver_parameters):

        mesh, facet_marking_f, volume_marking_f, tags = load_mesh(mesh_path)

        cell = mesh.ufl_cell()
        V = FunctionSpace(mesh, FiniteElement('CG', cell, 1))

        u = TrialFunction(V)
        v = TestFunction(V)

        soma, axon, dendrite = {1}, tags['axon'], tags['dendrite']
        assert not tags['axon']
        assert not tags['dendrite']

        insulated_surfaces = tags['probe_surfaces']
        grounded_surfaces = {5, 6}  # Enforced stongly
    
        # We now want to add all but the stimulated probe site to insulated
        # surfaces and for the stimulated probe prescribe ...
        stimulated_site = problem_parameters['stimulated_site']
        all_sites = tags['contact_surfaces']
        assert stimulated_site in all_sites

        all_sites.remove(stimulated_site)
        insulated_surfaces.update(all_sites)  # Rest is insulated

        # NOTE: leaving out insulated mean enforcing ewakly n.grad(u) = 0 

        # Add the stimulated site
        site_current = problem_parameters['site_current']
        ds = Measure('ds', subdomain_data=facet_marking_f)

        # Integrate over entire volume
        dx = Measure('dx')

        # cond_int = Constant(problem_parameters['cond_int'])
        cond_ext = Constant(problem_parameters['cond_ext'])

        a = inner(cond_ext*grad(u), grad(v))*dx
        L = inner(Constant(-site_current(0, 0, 0)[0]), v)*ds(stimulated_site)

        bc_grounded = [DirichletBC(V, Constant(0), facet_marking_f, tag)
                       for tag in grounded_surfaces]
        # Get the linear system
        assembler = SystemAssembler(a, L, bcs=bc_grounded)
        A, b = PETScMatrix(), PETScVector()
        assembler.assemble(A) 
        assembler.assemble(b)

        # Use iterative solver here to get it a bit faster
        # solver = PETScKrylovSolver('cg', 'hypre_amg')
        solver = PETScKrylovSolver('cg', 'petsc_amg')
        solver.parameters['relative_tolerance'] = 1E-12
        solver.parameters['absolute_tolerance'] = 1E-14
        solver.parameters['monitor_convergence'] = True
        solver.set_operators(A, A)
    
        uh = Function(V)
        x = as_backend_type(uh.vector())

        # The final thing is to get point sources - init with zeto and await
        # values
        point_sources = [np.fromiter(p, dtype=float)
                         for p in problem_parameters.get('point_sources', [])]

        # What to remember
        self.assembler = assembler
        self.solver = solver
        self.uh = uh
        self.b = b
        self.x = x
        self.point_sources = point_sources
        self.system_size = A.size(0)

        
    def __call__(self, point_source_values=None):
        '''Solve with new values'''

        self.assembler.assemble(self.b)
        # Invalidate point sources?        
        if point_source_values is None:
            self.solver.solve(self.x, self.b)
            return self.uh

        # With sources
        assert len(point_source_values) == len(self.point_sources)
        # that have magniture
        point_sources = [PointSource(self.uh.function_space(), Point(p), v)
                         for (p, v) in zip(self.point_sources, point_source_values)]
        # Create
        [p.apply(self.b) for p in point_sources]
        
        self.solver.solve(self.x, self.b)

        return self.uh
