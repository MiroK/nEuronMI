import dolfin as df


def mpi_comm():
    '''MPI world comm'''
    try:
        return df.MPI.comm_world
    except AttributeError:
        return df.mpi_comm_world()        
