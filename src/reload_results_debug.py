from dolfin import *
import sys
import os
from os.path import join
import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
import yaml
from copy import copy

if __name__ == '__main__':

    if '-f' in sys.argv:
        pos = sys.argv.index('-f')
        file = sys.argv[pos + 1]
    else:
        raise Exception('Pass result file as -f argument')

    if 'noprobe' in file:
        no_mesh = file
        w_mesh = file[:no_mesh.find('noprobe')] + 'wprobe'
    elif 'wprobe' in file:
        w_mesh = file
        no_mesh = file[:w_mesh.find('wprobe')] + 'noprobe'

    no_mesh = w_mesh
    
    comm = mpi_comm_world()
    # parameters['allow_extrapolation'] = True

    # open params file wmesh
    with open(join(w_mesh, 'params.yaml'), 'r') as f:
        info_w = yaml.load(f)

    # open params file woutmesh
    #with open(join(no_mesh, 'params.yaml'), 'r') as f:
    #    info_no = yaml.load(f)

    # find min idx
    v_p = np.load(join(file, 'v_probe.npy'))
    min_idx = np.unravel_index(v_p.argmin(), v_p.shape)[1]

    # u_ext mesh
    extra_mesh_path_w = info_w['mesh']['path']
    print extra_mesh_path_w
    extra_mesh_path_no = extra_mesh_path_w
    # neuron_mesh_path = join(file, 'neuron_mesh.h5')

    extra_mesh_w = Mesh()
    extra_mesh_no = Mesh()
    # neuron_mesh = Mesh()

    # Load meshes
    with HDF5File(comm, extra_mesh_path_w, 'r') as in_file:
        in_file.read(extra_mesh_w, 'mesh', False)
    with HDF5File(comm, extra_mesh_path_no, 'r') as in_file:
        in_file.read(extra_mesh_no, 'mesh', False)

    ext_w = []
    ext_no = []

    # curr = []
    conv = 1e-4
    x_axis = np.arange(11, 80)*conv
    z_axis = np.arange(-100, 100)*conv

    # Checkout the content of h5 file with `h5ls -r FILE.h5`
    with HDF5File(comm, join(w_mesh, 'u_sol.h5'), 'r') as in_file:
        # Now recreate a function space on iter
        V_u = FunctionSpace(extra_mesh_w, 'DG', 0)
        v_u = Function(V_u)  # All 0

        # for i in range(500):
        i = min_idx
        in_file.read(v_u.vector(), '/VisualisationVector/%d' % i, False)

        true = interpolate(Expression('A*(x[0]+x[1]+x[1])', A=i+1, degree=1), V_u)

        # print [v_u(x, 0, 0) for x in range(20, 100, 10)]

        ext_img = np.zeros((len(x_axis), len(z_axis)))

        for i, x_i in enumerate(x_axis):
            for j, z_j in enumerate(z_axis):
                try:
                    value = v_u(x_i, 0, z_j)
                except RuntimeError:
                    value = np.nan
                ext_img[i, j] = value
                print value
        ext_w.append(ext_img)

    # Checkout the content of h5 file with `h5ls -r FILE.h5`
    with HDF5File(comm, join(no_mesh, 'u_sol.h5'), 'r') as in_file:
        # for i in range(500):
        # Now recreate a function space on iter
        V_u_n = FunctionSpace(extra_mesh_no, 'DG', 0)
        v_u_n = Function(V_u_n)  # All 0

        i = min_idx
        in_file.read(v_u_n.vector(), '/VisualisationVector/%d' % i, False)

        true = interpolate(Expression('A*(x[0]+x[1]+x[1])', A=i + 1, degree=1), V_u_n)

        ext_img = np.zeros((len(x_axis), len(z_axis)))


        for i, x_i in enumerate(x_axis):
            for j, z_j in enumerate(z_axis):
                try:
                    value = v_u(x_i, 0, z_j)
                except RuntimeError:
                    value = np.nan
                ext_img[i, j] = value

        ext_no.append(ext_img)

    ext_w = np.squeeze(np.array(ext_w))
    ext_no = np.squeeze(np.array(ext_no))

    fig = plt.figure() #figsize=(4,15))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    minv = np.min([np.min(ext_no), np.min(ext_w)])
    maxv = np.max([np.max(ext_no), np.max(ext_w)])

    import matplotlib
    
    masked_array = np.ma.array(ext_w.T, mask=np.isnan(ext_w.T))
    cmap = matplotlib.cm.jet
    cmap.set_bad('white', 1.)

    ax1.matshow(masked_array, cmap=cmap, origin='lower', extent=[np.min(x_axis), np.max(x_axis),
                                                                 np.min(z_axis), np.max(z_axis)])
    #ax2.matshow(ext_no.T, cmap = 'jet', origin = 'lower', extent=[np.min(x_axis), np.max(x_axis),
    #                                                         np.min(z_axis), np.max(z_axis)],
    #            vmin=minv, vmax=maxv)

    probe_x = 40.0*conv
    probe_y = -100*conv

    ax1.add_patch(patches.Rectangle((probe_x, probe_y), 15*conv, 600*conv, fill=False, hatch='\\'))
    ax2.add_patch(patches.Rectangle((probe_x, probe_y), 15, 600, fill=False, hatch='\\'))

    ax1.axis('off')
    ax2.axis('off')

    plt.show()
    # assert (v.vector() - true.vector()).norm('linf') < 1E-15

    # # Load neuron currents
    # with HDF5File(comm, neuron_mesh_path, 'r') as in_file:
    #     in_file.read(neuron_mesh, 'mesh', False)
    #
    # # Now recreate a function space on iter
    # V_c = FunctionSpace(neuron_mesh, 'DG', 0)
    # v_c = Function(V_c)  # All 0
    #
    # # Checkout the content of h5 file with `h5ls -r FILE.h5`
    # with HDF5File(comm, join(file, 'current_sol.h5'), 'r') as in_file:
    #     for i in range(500):
    #         in_file.read(v_c.vector(), '/VisualisationVector/%d' % i, False)
    #
    #         true = interpolate(Expression('A*(x[0]+x[1]+x[1])', A=i + 1, degree=1), V_c)
    #         curr.append(copy(v_c))
    #         # assert (v.vector() - true.vector()).norm('linf') < 1E-15

    # plot x-z cross section
    # grid = np.zeros()



