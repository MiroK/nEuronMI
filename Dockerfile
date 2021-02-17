FROM quay.io/fenicsproject/stable:2017.2.0

ENV FENICS_VERSION="1:2017.2.0.1" 
ENV GMSH_VERSION="gmsh-4.4.1"
ENV GMSH_FOLDER=""$GMSH_VERSION"-Linux64-sdk"
ENV GMSH_URL="http://gmsh.info/bin/Linux/"$GMSH_FOLDER".tgz"
ENV PYTHONPATH=""

USER root

RUN apt-get update && \
    apt install -y git && \
    apt install -y mercurial && \
    apt-get install -y wget && \
    apt-get install -y software-properties-common && \
    apt-get install -y libglu1 libxft2 libxcursor-dev libxinerama-dev && \
    apt-get install -y libx11-dev bison flex automake libtool libxext-dev libncurses-dev python3-dev xfonts-100dpi cython3 libopenmpi-dev python3-scipy make zlib1g-dev
    
USER fenics

# Get Gmsh
RUN mkdir gmsh && \
    cd gmsh && \
    wget $GMSH_URL && \
    ls && \
    echo ""$GMSH_VERSION"-Linux64-sdk.tgz" && \
    tar xvzf ""$GMSH_VERSION"-Linux64-sdk.tgz" && \
    rm ""$GMSH_VERSION"-Linux64-sdk.tgz" 

ENV PYTHONPATH="/home/fenics/gmsh/$GMSH_FOLDER/lib":"$PYTHONPATH"

# Get NEURON
RUN mkdir neuron && \
    git clone https://github.com/neuronsimulator/nrn.git  && \
    cd nrn  && \
    git checkout 7.7.0  && \
    echo "Installing NEURON:"  && \
    echo "running sh build.sh"  && \
    sh build.sh > /dev/null  && \
    echo "running ./configure"  && \
    ./configure --prefix=$HOME/.local/nrn --without-iv --with-nrnpython=python --with-mpi --disable-rx3d  && \
    echo "running make"  && \
    make -j8  && \
    echo "running make install" && \
    make install && \
    cd src/nrnpython && \
    echo "installing neuron python module"  && \
    python setup.py install --user

# install LFPy
RUN pip install --user LFPy

# install MEAutility
RUN pip install --user MEAutility

# Install cbcbeat
RUN pip install git+https://bitbucket.org/dolfin-adjoint/pyadjoint.git@2019.1.0 --user && \
    pip install hg+https://bitbucket.org/meg/cbcbeat@2017.2.0 --user && \
    pip install cppimport --user && \
    pip install git+https://github.com/mikaem/fenicstools.git@2016.1 --user

# Get neuronmi
RUN git clone -b js https://github.com/MiroK/nEuronMI.git

# Generate cell models
#RUN cd nEuronMI/neuronmi/simulators/solver && \
#    /home/fenics/.local/bin/gotran2beat Hodgkin_Huxley_1952.ode && \
#    /home/fenics/.local/bin/gotran2beat Passive.ode && \    
#    cd

# Install neuronmi
RUN cd nEuronMI && \
    python setup.py develop --user && \
    cd ..

USER root

# Release
# Might need to do docker login 
# docker build --no-cache -t neuronmi .
# docker tag neuronmi:latest mirok/neuronmi
# docker push mirok/neuronmi
