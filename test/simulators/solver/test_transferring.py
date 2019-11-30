from neuronmi.simulators.solver.embedding import EmbeddedMesh
from neuronmi.simulators.solver.transferring import SubMeshTransfer
import dolfin as df
import numpy as np
import unittest


class TestCases(unittest.TestCase):

    def test_to_DG0(self):
        subdomains = (df.CompiledSubDomain('near(x[0], 0.5)'), df.DomainBoundary())
        
        for subd in subdomains: 
            mesh = df.UnitCubeMesh(4, 4, 4)
            facet_f = df.MeshFunction('size_t', mesh, 2, 0)
            subd.mark(facet_f, 1)
                
            submesh = EmbeddedMesh(facet_f, 1)

            transfer = SubMeshTransfer(mesh, submesh)

            V = df.FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
            Vsub = df.FunctionSpace(submesh, 'DG', 0)
                
            to_Vsub = transfer.compute_map(Vsub, V, strict=False)
            # Set degree 0 to get the quad order right
            f = df.Expression('x[0] + 2*x[1] - x[2]', degree=0)
                
            fV = df.interpolate(f, V)
            fsub = df.Function(Vsub)
                
            to_Vsub(fsub, fV)
                
            error = df.inner(fsub - f, fsub - f)*df.dx(domain=submesh)
            error = df.sqrt(abs(df.assemble(error)))
                
            self.assertTrue(error < 1E-13)

    def test_from_DG0(self):
        
        def main(n, chi, measure, restrict):
            mesh = df.UnitCubeMesh(2, n, n)
            facet_f = df.MeshFunction('size_t', mesh, 2, 0)
            chi.mark(facet_f, 1)

            submesh = EmbeddedMesh(facet_f, 1)
            
            transfer = SubMeshTransfer(mesh, submesh)

            V = df.FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
            Vsub = df.FunctionSpace(submesh, 'DG', 0)

            to_V = transfer.compute_map(V, Vsub, strict=False)
            # Set degree 0 to get the quad order right
            f = df.Expression('x[0] + 2*x[1] - x[2]', degree=0)
            
            fsub = df.interpolate(f, Vsub)
            fV = df.Function(V)

            to_V(fV, fsub)

            y = V.tabulate_dof_coordinates().reshape((V.dim(), -1))
            x = Vsub.tabulate_dof_coordinates().reshape((Vsub.dim(), -1))
            # Correspondence of coordinates
            self.assertTrue(np.linalg.norm(y[transfer.cache] - x) < 1E-13)
            # These are all the coordinates
            idx = list(set(range(V.dim())) - set(transfer.cache))
            self.assertTrue(not any(chi.inside(xi, True) for xi in y[idx]))

            dS_ = df.Measure(measure, domain=mesh, subdomain_data=facet_f)
            # Stange that this is not exact
            error = df.inner(restrict(fV - f), restrict(fV - f))*dS_(1)
            error = df.sqrt(abs(df.assemble(error, form_compiler_parameters={'quadrature_degree': 0})))
            
            return error

        # Trick domains boundary beacuse we want to use inside properly?
        bdry = df.CompiledSubDomain(' || '.join(['near(x[0]*(1-x[0]), 0)',
                                                 'near(x[1]*(1-x[1]), 0)',
                                                 'near(x[2]*(1-x[2]), 0)']))

        inputs = ((df.CompiledSubDomain('near(x[0], 0.5)'), 'dS', df.avg),
                  (bdry, 'ds', lambda x: x))
        for chi, measure, restrict in inputs:
            errors = []
            for n in (2, 4, 8, 16):
                e = main(n, chi, measure, restrict)
                self.assertTrue(not errors or e < errors[-1])
                errors.append(e)

    def test_to_DG0_subdomain(self):
        mesh = df.UnitSquareMesh(4, 4)
        cell_f = df.MeshFunction('size_t', mesh, 2, 0)
        df.CompiledSubDomain('x[0] < 0.5 + DOLFIN_EPS').mark(cell_f, 1)

        submesh = EmbeddedMesh(cell_f, 1)

        transfer = SubMeshTransfer(mesh, submesh)

        V = df.FunctionSpace(mesh, 'DG', 0)
        Vsub = df.FunctionSpace(submesh, 'DG', 0)

        to_Vsub = transfer.compute_map(Vsub, V, strict=True)
        # Set degree 0 to get the quad order right
        f = df.Expression('x[0] + 2*x[1]', degree=0)

        fV = df.interpolate(f, V)
        fsub = df.Function(Vsub)

        to_Vsub(fsub, fV)

        error = df.inner(fsub - f, fsub - f)*df.dx(domain=submesh)
        error = df.sqrt(abs(df.assemble(error)))

        self.assertTrue(error < 1E-13)

    def test_from_DG0_subdomain(self):
        mesh = df.UnitSquareMesh(4, 4)
        cell_f = df.MeshFunction('size_t', mesh, 2, 0)
        df.CompiledSubDomain('x[0] < 0.5 + DOLFIN_EPS').mark(cell_f, 1)

        submesh = EmbeddedMesh(cell_f, 1)

        transfer = SubMeshTransfer(mesh, submesh)

        V = df.FunctionSpace(mesh, 'DG', 0)
        Vsub = df.FunctionSpace(submesh, 'DG', 0)

        to_V = transfer.compute_map(V, Vsub, strict=True)
        # Set degree 0 to get the quad order right
        f = df.Expression('x[0] + 2*x[1]', degree=0)

        fsub = df.interpolate(f, Vsub)
        fV = df.Function(V)

        to_V(fV, fsub)

        error = df.inner(fV - f, fV - f)*df.dx(domain=submesh)
        error = df.sqrt(abs(df.assemble(error)))

        self.assertTrue(error < 1E-13)
