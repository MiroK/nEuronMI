Installation
=============


Using docker (recommended)
---------------------------

We recommend using our docker `container <https://hub.docker.com/r/mirok/neuronmi>`_
which has all the dependencies pre-installed.

You can run the docker image with:

.. code-block:: bash

    docker run mirok/neuronmi


Manual installation
--------------------

The following are dependencies of `neuronmi` and how they can be obtained.

**IMPORTANT**: the current version runs on Python 2.7. We are currently working on a Python 3 update.

1. Generating meshes for neuron simulations with EMI models
We rely on `Gmsh <http://gmsh.info/>`_ for both mesh generation and geometry definition.
All is done via the python `API <https://gitlab.onelab.info/gmsh/gmsh/blob/master/api/gmsh.py>`_ of Gmsh.
The gmsh module has to be on python path. For the current shell session
this can be accomplished by running

.. code-block:: bash

    export PYTHONPATH=`pwd`:"$PYTHONPATH"

in the directory where `gmsh.py` resides (e.g. /usr/local/lib/).

2. Partial differential equation part of EMI
The solver requires `FEniCS <https://fenicsproject.org/download/>`_ version 2017.2.0. In our
experience the simplest way of installation that also plays along nicely with Gmsh is by
using the dedicated Ubuntu `package <https://packages.ubuntu.com/bionic/math/fenics>`_.

3. Ordinary differential equation part of EMI
Membrane physics is solved for using `cbc.beat <https://bitbucket.org/meg/cbcbeat>`_
(which depends on `dolfin-adjoint <http://dolfin-adjoint-doc.readthedocs.io/en/latest/download/index.html>`_).

Testing
--------

Run from current directory

.. code-block:: bash

    python -m unittest discover ./test/mesh;
    python -m unittest discover ./test/simulators/solver;

