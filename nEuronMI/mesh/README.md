# Generating meshes for neuron simulations with EMI models

We rely on [Gmsh](http://gmsh.info/) for both mesh generation and geometry defition.
All is done via python [API](https://gitlab.onelab.info/gmsh/gmsh/blob/master/api/gmsh.py) of Gmsh.
The gmsh module has to be on python path. For the current shell session
this can be accomplised by running `export PYTHONPATH=`pwd`:"$PYTHONPATH"`
in the directory where **gmsh.py** resides (e.g. /usr/local/lib/).