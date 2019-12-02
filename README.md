# FEM simulator for realistic neuron models and probes

### Testing

Run from current directory
```python
python -m unittest discover ./test
```

### Dependencies
#### 1. Generating meshes for neuron simulations with EMI models
We rely on [Gmsh](http://gmsh.info/) for both mesh generation and geometry defition.
All is done via python [API](https://gitlab.onelab.info/gmsh/gmsh/blob/master/api/gmsh.py) of Gmsh.
The gmsh module has to be on python path. For the current shell session
this can be accomplised by running 

```bash
export PYTHONPATH=`pwd`:"$PYTHONPATH"
```

in the directory where **gmsh.py** resides (e.g. /usr/local/lib/).

#### 2. Partial differential equation part of EMI
The solver requires [FEniCS](https://fenicsproject.org/download/) version 2017.2.0. In our 
experience the simplest way of installation that also plays along nicely with Gmsh is by 
using the dedicated Ubuntu [package](https://packages.ubuntu.com/bionic/math/fenics)

#### 3. Ordinary differential equation part of EMI
Membrane physics is solved for using [cbc.beat](https://bitbucket.org/meg/cbcbeat)
  (which depends on [dolfin-adjoint](http://dolfin-adjoint-doc.readthedocs.io/en/latest/download/index.html))

#### Optional
An optional dependency for computing the probe for contact surfaces is [networkx](https://networkx.github.io/)
