# FEM simulator for relaistic neuron models and probes

### Dependencies
- mesh generation relies on [Gmsh](https://gmsh.info/#Download)
- the solver requires [FEniCS](https://fenicsproject.org/download/), [cbc.beat](https://bitbucket.org/meg/cbcbeat)
  (which depends on [dolfin-adjoint](http://dolfin-adjoint-doc.readthedocs.io/en/latest/download/index.html))
- the current code has been tested with FEniCS 2017.2.0 (fetched as the Ubuntu package),
  the other dependencies were built from source again the latest maste (18/01/2018)
  branches.
- other (for now) optional dependencies are [networkx](https://networkx.github.io/) and [fenicstools](https://github.com/mikaem/fenicstools)

### Create mesh

- in `mesh/geometries` change parameters of `geogen.py` and run (add name to generated file).

- open the generated file in gmsh and refine by splitting + optimize 3D
  Splitting may be done several time until the desired resolutions is met.
  In this respect, the `*_mesh_size` parameters in `geogen.py` determine
  the ratios of element sizes in the given region. After refinement keep
  an eye on the number of elements (Tools->Statistics).
  
- save the .msh file

- run `msh_convert.py` with the .msh file as argument to create the .h5 file used from FEniCS

### Solve the FEM problem
- see `example.py`
