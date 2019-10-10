from dolfin import *

mesh = UnitSquareMesh(32, 32)
cell_f = CellFunction('size_t', mesh, 2)

inside = ' && '.join(['0.25-tol<x[0]', 'x[0]<0.75+tol', '0.25-tol<x[1]', 'x[1]<0.75+tol'])
CompiledSubDomain(inside, tol=1e-13).mark(cell_f, 1)

Sel = FiniteElement('RT', triangle, 1)
Uel = FiniteElement('DG', triangle, 0)
Qel = FiniteElement('Discontinuous Lagrange Trace', triangle, 0)
Wel = MixedElement([Sel, Uel, Qel])

W = FunctionSpace(mesh, Wel)
w = Function(W)

x = SpatialCoordinate(mesh)
w = interpolate(Constant((1, 2, 3, 4)), W)

facet_f = FacetFunction('size_t', mesh, 0)
mesh.init(1)
mesh.init(1, 2)
for f in facets(mesh):
    if len(f.entities(2)) == 2:
        if len(set(cell_f[int(e)] for e in f.entities(2))) == 2:
            facet_f[f] = 1

dx = Measure('dx', domain=mesh, subdomain_data=cell_f)
dS = Measure('dS', domain=mesh, subdomain_data=facet_f)
    
n = FacetNormal(mesh)('-')


            
from embedding import EmbeddedMesh
bmesh, _ = EmbeddedMesh(facet_f, [1])

Q = FunctionSpace(mesh, Qel)
Qb = FunctionSpace(bmesh, 'DG', 0)

from transferring import SubMeshTransfer
transfer = SubMeshTransfer(mesh, bmesh)

toQ_fromW2 = FunctionAssigner(Q, W.sub(2))
assign_toQ_neuron_fromQ = transfer.compute_map(Qb, Q, strict=False)

tau, v, q = TestFunctions(W)
current_form = 1./FacetArea(mesh)('+')*inner(dot(w.sub(0)('+'), n), q('+'))*dS(1)
current_form += inner(Constant(0), v)*dx(2)  # Fancy zero for orientation

current_out, current_aux = Function(Qb), Function(Q)

w_aux = Function(W)
w_aux.vector()[:] = assemble(current_form)
toQ_fromW2.assign(current_aux, w_aux.sub(2))
assign_toQ_neuron_fromQ(current_out, current_aux)

for cell in cells(bmesh):
    x, y = cell.midpoint()[0], cell.midpoint()[1]
    normal = current_out(x, y)
    if near(y, 0.75):  # (1, 2).(0, 1)
        assert near(normal, 2, 1E-10)
    if near(y, 0.25):  # (1, 2).(0, -1)
        assert near(normal, -2, 1E-10), normal
    if near(x, 0.75):  # (1, 2).(1, 0)
        assert near(normal, 1, 1E-10)
    if near(x, 0.25):  # (1, 2).(-1, 0)
        assert near(normal, -1, 1E-10)
