import cbcbeat as beat
from Hodgkin_Huxley_1952 import Hodgkin_Huxley_1952
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

    # Adjust stimulus current
    # Note: Stimulation is currently implemented as a part of the passive membrane model
    # and given on the form: I_s = g_s(x)*exp(-t/alpha)(v-v_eq)
    dendrite_params["alpha"] = 2.0  # (ms)
    dendrite_params["v_eq"] = 0.0   # (ms)
    dendrite_params["g_s"] = Expression("stim_strength*(x[2]>1.0)",
                                        stim_strength=problem_parameters["stim_strength"],
                                                                            degree=1)

    # Update dendrite parameters
    dendrite_model = Passive(dendrite_params)

    # Set up ode solver parameters
    Solver = beat.BasicCardiacODESolver
    odesolver_params = Solver.default_parameters()
    odesolver_params["theta"] = 0.5    # Crank-Nicolson
    
    # Generate membrane model with multiple ode models
    neuron_model = beat.MultiCellModel((soma_model, axon_model, dendrite_model),
                                       (soma, axon, dendrite),
                                       subdomains)

    mesh = subdomains.mesh()

    ode_solver = beat.BasicCardiacODESolver(mesh, time, neuron_model, I_s=Constant(0.0),
                                            params=odesolver_params)
                                            

    return ode_solver, neuron_model
