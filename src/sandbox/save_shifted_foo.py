from dolfin import *
import numpy as np


def shift_function(f, shift):
    mesh = Mesh(f.function_space().mesh())
    mesh.coordinates()[:] += shift
    V = FunctionSpace(mesh, f.function_space().ufl_element())
    return Function(V, f.vector().copy())


mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, 'CG', 1)

shift = np.array([0.5, 0.5])

# Original
f = interpolate(Expression('x[0]*x[0] + x[1]*x[1]', degree=1), V)

# Lambda which can be evaluated but not stored
f_shift = lambda x, f=f, shift=shift: f(x-shift)

# Something that could be shoter and is the same
F = shift_function(f, shift)

print [abs(f_shift(p) - F(p)) for p in ([0.6, 0.7],
                                        [0.7, 0.6321],
                                        [0.7, 0.64735],
                                        [0.7345, 0.87],
                                        [0.6545, 0.8787])]


