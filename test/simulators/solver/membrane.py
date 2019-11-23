from dolfin import *
from neuronmi.simulators.solver.Hodgkin_Huxley_1952 import Hodgkin_Huxley_1952
from neuronmi.simulators.solver.Passive import Passive
from neuronmi.simulators.solver.membrane import *


mesh = UnitSquareMesh(32, 32)
cell_f = MeshFunction('size_t', mesh, mesh.topology().dim(), 2)

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
                                    stim_strength=1E-2,
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

gen = solver.solve((0, 1), 1E-3)
f = File('foo.pvd')
for ((t0, t1), x) in gen:
    print(x, '>>>>>>>>>>>>>>>>>>>>>>>>>>>>', t1, x.vector().norm('l2'))
    f << x, t1

    x.vector()[:] += 1.
