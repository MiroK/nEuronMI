import cbcbeat as beat
from Hodgkin_Huxley_1952 import Hodgkin_Huxley_1952
from transferring import SubMeshTransfer
from aux import subdomain_bbox
from Passive import Passive
from dolfin import *


def ODESolver(subdomains, soma, axon, dendrite, problem_parameters):
    '''
    Setup a membrane model and an ODE solver for it.
    '''
    
    time = Constant(0.0)  # Start from now
    
    # Assign membrane models to soma, axon and dendirite
    soma_model = Hodgkin_Huxley_1952()
    axon_model = Hodgkin_Huxley_1952()
    dendrite_model = Passive()
    
    # Adjust parameters of the dendrite model
    # Note: similar adjustments may be done for the soma and axon models in the same manner
    dendrite_params = dendrite_model.default_parameters()
    dendrite_params["g_leak"] = 0.06    #  passive membrane conductance (in mS/cm**2)
    dendrite_params["E_leak"] = -75.0   #  passive resting membrane potential (in mV)
    dendrite_params["Cm"] = 1.0         #  membrane capacitance (in uF/cm**2)

    # Extract the bounds of the z coordinate to localize stimulation
    zmin, zmax = subdomain_bbox(subdomains)[-1]
    # Or just for the dendrite part
    zmin_dend, zmax_dend = subdomain_bbox(subdomains, dendrite)[-1]

    # Adjust stimulus current
    # Note: Stimulation is currently implemented as a part of the passive membrane model
    # and given on the form: I_s = g_s(x)*exp(-t/alpha)(v-v_eq)
    dendrite_params["alpha"] = 2.0  # (ms)
    dendrite_params["v_eq"] = 0.0   # (mV)
    dendrite_params["t0"] =  problem_parameters["stim_start"]  # (ms)
    dendrite_params["g_s"] = Expression("stim_strength*(x[2]>1.0)",
                                        stim_strength=problem_parameters["stim_strength"],
                                                                            degree=1)

    # Update dendrite parameters
    dendrite_model = Passive(dendrite_params)

    # Set up ode solver parameters - to be used for all the subdomain
    Solver = beat.BasicCardiacODESolver
    odesolver_params = Solver.default_parameters()
    odesolver_params["theta"] = 0.5    # Crank-Nicolson
    # The timer in adjoint causes trouble so disable for now
    odesolver_params['enable_adjoint'] = False
    
    ode_solver = SubDomainCardiacODESolver(subdomains,
                                           models={soma: soma_model,
                                                   dendrite: dendrite_model,
                                                   axon: axon_model},
                                           ode_solver=Solver,
                                           params=odesolver_params)

    
    return ode_solver


class SubDomainCardiacODESolver(object):
    '''
    This class is effectively a container for several ODE models defined
    on the subdomains. However it presents the user with just one potential
    s.t. potential(on subdomain J) = potential of J'th model.
    '''
    def __init__(self, subdomains, models, ode_solver, params):
        time = Constant(0)
        # The subdomain solver instances
        tags = models.keys()
        models = [models[tag] for tag in tags]

        mesh = subdomains.mesh()
        submeshes = [SubMesh(mesh, subdomains, tag) for tag in tags]
        solvers = [ode_solver(submesh, time, model, I_s=Constant(0.0), params=params)
                   for submesh, model in zip(submeshes, models)]

        # Set up initial conditions
        for solver, model in zip(solvers, models):
            (ode_solution0, _) = solver.solution_fields()
            ode_solution0.assign(model.initial_conditions())

        # It's mandatory that each subdomain has potential in the same space
        V_elm = set(solver.VS.sub(0).ufl_element() for solver in solvers)
        assert len(V_elm) == 1
        V_elm = V_elm.pop()

        # What we will show to the world
        V = FunctionSpace(mesh, V_elm)

        transfers = [SubMeshTransfer(mesh, submesh) for submesh in submeshes]

        self.V = V
        self.solvers = solvers
        self.transfers = transfers
        
    def solve(self, interval, dt=None):
        # MixedODE(sub) <--> PotentialODE(sub) <---> FullPotential
        adapters = []
        toSub_fromMixed_map, toWhole_fromSub_map = [], []
        toSub_fromWhole_map, toMixed_fromSub_map = [], []
        for transfer, solver in zip(self.transfers, self.solvers):
            W = solver.vs.function_space() 
            Vsub = W.sub(0).collapse()
            
            adapters.append(Function(Vsub))

            toSub_fromMixed_map.append(FunctionAssigner(Vsub, W.sub(0)))
            toWhole_fromSub_map.append(transfer.compute_map(self.V, Vsub))

            toSub_fromWhole_map.append(transfer.compute_map(Vsub, self.V))
            toMixed_fromSub_map.append(FunctionAssigner(W.sub(0), Vsub))

        # Init all the ODE solver
        generators = [solver.solve(interval, dt) for solver in self.solvers]

        v_whole = Function(self.V)
        solutions = []

        stopped = [False]*len(generators)
        while True: 
            # Step all
            solutions = []
            for j, gen in enumerate(generators):
                try:
                    (t0, t1), uj = next(gen)
                except StopIteration:
                    stopped[j] = True
                solutions.append(uj)
                
            if all(stopped): raise StopIteration
            # Glue
            for j, uj in enumerate(solutions):
                toSub_fromMixed_map[j].assign(adapters[j], uj.sub(0))
                toWhole_fromSub_map[j](v_whole, adapters[j])
                
            # Expose 
            yield (t0, t1), v_whole

            # Put updated whole back to the solvers. The idea is that
            # outside v_whole is modified
            for j, uj in enumerate(solutions):
                toSub_fromWhole_map[j](adapters[j], v_whole)
                toMixed_fromSub_map[j].assign(uj.sub(0), adapters[j])

            # They either all be true or all be False
            assert len(set(stopped)) == 1

            
# --------------------------------------------------------------------


if __name__ == '__main__':
    mesh = UnitSquareMesh(32, 32)
    cell_f = CellFunction('size_t', mesh, 2)

    inside = ' && '.join(['0.25-tol<x[0]', 'x[0]<0.75+tol', '0.25-tol<x[1]', 'x[1]<0.75+tol'])
    CompiledSubDomain(inside, tol=1e-13).mark(cell_f, 1)

    axon_model = Hodgkin_Huxley_1952()

    dendrite_model = Passive()
    
    # Adjust parameters of the dendrite model
    # Note: similar adjustments may be done for the soma and axon models in the same manner
    dendrite_params = dendrite_model.default_parameters()
    dendrite_params["g_leak"] = 0.06    #  passive membrane conductance (in mS/cm**2)
    dendrite_params["E_leak"] = -75.0   #  passive resting membrane potential (in mV)
    dendrite_params["Cm"] = 1.0         #  membrane capacitance (in uF/cm**2)

    # Adjust stimulus current
    # Note: Stimulation is currently implemented as a part of the passive membrane model
    # and given on the form: I_s = g_s(x)*exp(-t/alpha)(v-v_eq)
    dendrite_params["alpha"] = 2.0  # (ms)
    dendrite_params["v_eq"] = 0.0   # (mV)
    dendrite_params["g_s"] = Expression("stim_strength*(x[2]>1.0)",
                                        stim_strength=1.0,
                                        degree=1)

    # Update dendrite parameters
    dendrite_model = Passive(dendrite_params)

    Solver = beat.BasicCardiacODESolver
    odesolver_params = Solver.default_parameters()
    # The timer in adjoint causes trouble so disable for now
    odesolver_params['enable_adjoint'] = False

    solver = SubDomainCardiacODESolver(subdomains=cell_f,
                                       models={1:dendrite_model, 2: axon_model},
                                       ode_solver=Solver,
                                       params=odesolver_params)

    gen = solver.solve((0, 1), 1E-1)
    f = File('foo.pvd')
    for ((t0, t1), x) in gen:
        print x, '>>>>>>>>>>>>>>>>>>>>>>>>>>>>', t1, x.vector().norm('l2')
        f << x, t1

        x.vector()[:] += 1.
