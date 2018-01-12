# FEM simulator for relaistic neuron models and probes

### Create mesh

- in `mesh/geometries` change parameters of `geogen.py` and run (add name to generated file).

- open the generated file in mshr and refine by splitting + optimize 3D

- save the .msh file

- run `msh_convert.py` with the .msh file as argument to create the .h5 file used from FEniCS


## Solve the FEM problem
