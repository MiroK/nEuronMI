
"""This module contains a Hodgkin_Huxley_1952 cardiac cell model

The module was autogenerated from a gotran ode file
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
        params = OrderedDict([("g_L", 0.06),
                              ("E_L", -75.0),
                              ("Cm", 1.0),
                              ("g_S", 0.0),
                              ("alpha", 2.0),
                              ("v_eq", 0.0),
                              ("t0", 0.0)])
        return params

    @staticmethod
    def default_initial_conditions():
        "Set-up and return default initial conditions."
        ic = OrderedDict([("V", -75.0), ("S", 0.0)])
        return ic

    def _I(self, v, s, time):
        """
        Original gotran transmembrane current dV/dt
        """
        time = time if time else Constant(0.0)

        # Assign states
        V = v

        # Assign parameters
        g_leak = self._parameters["g_L"]
        Cm = self._parameters["Cm"]
        E_leak = self._parameters["E_L"]
        g_s = self._parameters["g_S"]
        alpha = self._parameters["alpha"]
        v_eq = self._parameters["v_eq"]
        t0 = self._parameters["t0"]

        # Init return args
        current = [ufl.zero()]*1

        # Expressions for the Membrane component
        i_leak = g_leak*(-E_leak + V)
        i_stim = g_s*exp(-(time-t0)/alpha)*(V-v_eq)*ufl.conditional(ufl.ge(time, t0), 1, 0)
        current[0] = (-i_leak-i_stim)/Cm

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

        return 0.0

    def num_states(self):
        return 1

    def __str__(self):
        return 'Passive membrane model'
