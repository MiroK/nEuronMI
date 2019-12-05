FROM quay.io/fenicsproject/base:latest

# install fenics
ENV FENICS_VERSION="1:2017.2.0.1" 
ENV GMSH_VERSION="gmsh-4.4.1"
ENV GMSH_FOLDER=""$GMSH_VERSION"-Linux64-sdk"
ENV GMSH_URL="http://gmsh.info/bin/Linux/"$GMSH_FOLDER".tgz"
ENV PYTHONPATH=""

USER root

RUN apt-get update && \
    apt install -y python2.7 python-pip && \
    apt install -y git && \
    apt install -y mercurial && \
    apt-get install -y wget && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:fenics-packages/fenics && \
    apt-get update && \
    apt-get install -y --no-install-recommends fenics=$FENICS_VERSION && \
    apt-get install -y libglu1 libxft2 libxcursor-dev libxinerama-dev 
    
USER fenics

RUN mkdir gmsh && \
    cd gmsh && \
    wget $GMSH_URL && \
    ls && \
    echo ""$GMSH_VERSION"-Linux64-sdk.tgz" && \
    tar xvzf ""$GMSH_VERSION"-Linux64-sdk.tgz" && \
    rm ""$GMSH_VERSION"-Linux64-sdk.tgz" && \
    export PYTHONPATH="`pwd`/$GMSH_FOLDER/lib":"$PYTHONPATH" && \
    cd ..

# install cbcbeat
RUN pip install git+https://bitbucket.org/dolfin-adjoint/pyadjoint.git@2019.1.0 --user && \
    pip install hg+https://bitbucket.org/meg/cbcbeat --user

# clone nEuronMI
RUN git clone https://github.com/MiroK/nEuronMI.git && \
    cd nEuronMI && \
    git checkout debug && \
    cd ..

USER root

