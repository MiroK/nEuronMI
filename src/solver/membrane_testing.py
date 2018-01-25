import cbcbeat as beat
from Hodgkin_Huxley_1952 import Hodgkin_Huxley_1952
from Passive import Passive
from dolfin import *


def ODESolver(subdomains, soma, axon, dendrite, problem_parameters):
    '''
    Setup a membrane model and an ODE solver for it.
    '''
    
    time = Constant(0.0)  # Start from now
    
    multi_cell = True
    
    if multi_cell:
    
        # Assign membrane models to soma, axon and dendirite
        soma_model = Hodgkin_Huxley_1952()
        axon_model = Hodgkin_Huxley_1952()
        dendrite_model = Hodgkin_Huxley_1952()
    
        # Generate membrane model with multiple ode models
        neuron_model = beat.MultiCellModel((soma_model, axon_model, dendrite_model),
                                       (soma, axon, dendrite),
                                       subdomains)
    else:

        # Generate a single model on the entire membrane
        neuron_model = Hodgkin_Huxley_1952()

    # Set up ode solver parameters
    Solver = beat.BasicCardiacODESolver
    odesolver_params = Solver.default_parameters()
    odesolver_params["theta"] = 0.5    # Crank-Nicolson

    mesh = subdomains.mesh()

    ode_solver = beat.BasicCardiacODESolver(mesh, time, neuron_model, I_s=Constant(0.0),
                                            params=odesolver_params)
                                            

    return ode_solver, neuron_model
