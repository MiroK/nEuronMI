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
    apt-get install -y libglu1 libxft2 libxcursor-dev libxinerama-dev 
    
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

# Install cbcbeat
RUN pip install git+https://bitbucket.org/dolfin-adjoint/pyadjoint.git@2019.1.0 --user && \
    pip install hg+https://bitbucket.org/meg/cbcbeat@2017.2.0 --user && \
    pip install git+https://bitbucket.org/finsberg/gotran.git --user
    
# Get neuronmi
RUN git clone https://github.com/MiroK/nEuronMI.git

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
