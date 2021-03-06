# FEM simulator for realistic neuron models and probes

## Docker based installation installation (recommended)

We recommend using our docker [container](https://hub.docker.com/r/mirok/neuronmi)
which has all the dependencies preinstalled. 

You can pull the docker container with: 
```bash
docker pull mirok/neuronmi
```

And run the container with a shared folder in the current directory with:
```bash
docker run -ti -v $(pwd):/home/fenics/shared mirok/neuronmi
```

<!--- The image is used for testing the codebase with the current status [![CircleCI](https://circleci.com/gh/MiroK/nEuronMI.svg?style=svg)](https://circleci.com/gh/MiroK/nEuronMI). --->


## Manual installation 

The following are dependencies of `neuronmi` and how they can be obtained

#### 1. Generating meshes for neuron simulations with EMI models
We rely on [Gmsh](http://gmsh.info/) for both mesh generation and geometry defition.
All is done via python [API](https://gitlab.onelab.info/gmsh/gmsh/blob/master/api/gmsh.py) of Gmsh.
The gmsh module has to be on python path. For the current shell session
this can be accomplised by running 

```bash
export PYTHONPATH=`pwd`:"$PYTHONPATH"
```

in the directory where `gmsh.py` resides (e.g. /usr/local/lib/).

#### 2. Partial differential equation part of EMI
The solver requires [FEniCS](https://fenicsproject.org/download/) version 2017.2.0. In our 
experience the simplest way of installation that also plays along nicely with Gmsh is by 
using the dedicated Ubuntu [package](https://packages.ubuntu.com/bionic/math/fenics).

#### 3. Ordinary differential equation part of EMI
Membrane physics is solved for using [cbc.beat](https://bitbucket.org/meg/cbcbeat)
  (which depends on [dolfin-adjoint](http://dolfin-adjoint-doc.readthedocs.io/en/latest/download/index.html)).


### Testing

Run from current directory
```bash
python -m unittest discover ./test/mesh
python -m unittest discover ./test/simulators/solver
```

## Running the demo scripts

We provide two demo scripts to:

1. Simulate ephaptic effects between a pair of neurons

2. Simulate the effect of neural probes on extracellular action potentials

To run the demo scripts, go into the demo folder and run:

```bash
python ephaptic_effect.py

python probe_effect.py
```
