"""
This module contains a Passive cardiac cell model
"""
from __future__ import division
from collections import OrderedDict
import ufl

from cbcbeat.dolfinimport import *
from cbcbeat.cellmodels import CardiacCellModel

class Passive(CardiacCellModel):
    def __init__(self, params=None, init_conditions=None):
        """
        Create cardiac cell model

        *Arguments*
         params (dict, :py:class:`dolfin.Mesh`, optional)
           optional model parameters
         init_conditions (dict, :py:class:`dolfin.Mesh`, optional)
           optional initial conditions
        """
        CardiacCellModel.__init__(self, params, init_conditions)

    @staticmethod
    def default_parameters():
        "Set-up and return default parameters."
        params = OrderedDict([("Cm", 1),
                              ("E_leak", -75),
                              ("alpha", 2.0),
                              ("g_leak", 0.06),
                              ("g_s", 0.0),
                              ("t0", 0.0),
                              ("v_eq", 0.0)])
        return params

    @staticmethod
    def default_initial_conditions():
        "Set-up and return default initial conditions."
        ic = OrderedDict([("V", -75),
                          ("s", 0),
                          ("m", 1)])
        return ic

    def _I(self, v, s, time):
        """
        Original gotran transmembrane current dV/dt
        """
        time = time if time else Constant(0.0)

        # Assign states
        V = v
        assert(len(s) == 2)
        s, m = s

        # Assign parameters
        Cm = self._parameters["Cm"]
        E_leak = self._parameters["E_leak"]
        alpha = self._parameters["alpha"]
        g_leak = self._parameters["g_leak"]
        g_s = self._parameters["g_s"]
        t0 = self._parameters["t0"]
        v_eq = self._parameters["v_eq"]

        # Init return args
        current = [ufl.zero()]*1

        # Expressions for the Membrane component
        i_stim = g_s*(-v_eq + V)*ufl.conditional(ufl.ge(time, t0), 1,\
            0)*ufl.exp((t0 - time)/alpha)
        i_leak = g_leak*(-E_leak + V)
        current[0] = (-i_leak - i_stim)/Cm



        # Return results
        return current[0]

    def I(self, v, s, time=None):
        """
        Transmembrane current

           I = -dV/dt

        """
        return -self._I(v, s, time)

    def F(self, v, s, time=None):
        """
        Right hand side for ODE system
        """
        time = time if time else Constant(0.0)

        # Assign states
        V = v
        assert(len(s) == 2)
        s, m = s

        # Assign parameters

        # Init return args
        F_expressions = [ufl.zero()]*2

        # Expressions for the Membrane component
        F_expressions[0] = -m
        F_expressions[1] = s

        # Return results
        return dolfin.as_vector(F_expressions)

    def num_states(self):
        return 2

    def __str__(self):
        return 'Passive cardiac cell model'
