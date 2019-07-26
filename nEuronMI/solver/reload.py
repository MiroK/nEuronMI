from dolfin import *

mesh = UnitCubeMesh(10, 10, 10)
V = FunctionSpace(mesh, 'DG', 0)
f = Function(V)
    
comm = mpi_comm_world()
# Save functions
with XDMFFile(comm, 'test_reload.xdmf') as out_file:
    for i in range(10):
        f.assign(interpolate(Expression('A*(x[0]+x[1]+x[1])', A=i+1, degree=1), V))
        out_file.write(f, 0.1*i)  # Fake time step
            
# Now read back
# There seems to be a bug in loading mesh from function files so we
# load it instead from the original data
# Fakse stored mesh
with HDF5File(comm, 'test_reload_mesh.h5', 'w') as out_file:
    out_file.write(mesh, 'mesh')
        
mesh = Mesh()
# First get the mesh (Same for all time steps)
with HDF5File(comm, 'test_reload_mesh.h5', 'r') as in_file:
    in_file.read(mesh, 'mesh', False)
    
# Now recreate a function space on iter
V = FunctionSpace(mesh, 'DG', 0)
v = Function(V)  # All 0
# Fill with simulation data
# Checkout the content of h5 file with `h5ls -r FILE.h5`
with HDF5File(comm, 'test_reload.h5', 'r') as in_file:
    for i in range(10):
        in_file.read(v.vector(), '/VisualisationVector/%d' % i, False)

        true = interpolate(Expression('A*(x[0]+x[1]+x[1])', A=i+1, degree=1), V)
        assert (v.vector() - true.vector()).norm('linf') < 1E-15
