#!/usr/bin/env python

import os
from os.path import join
import sys
from glob import glob
import numpy as np
import json
# import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Qt5Agg')
# matplotlib.use('agg')
import pylab as plt
import LFPy

import mpl_toolkits.mplot3d as a3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.gridspec as gridspec
from matplotlib import colors as mpl_colors
from matplotlib.patches import Rectangle
from neuroplot import *
import MEAutility as mea
# from plotting_convention import *

save_fig = True
plot_3d = True

cell_folder = 'L5_TTPC1_cADpyr232_1'
cell_name = 'L5_TTPC1_cADpyr232_1'

MEAname = 'Neuronexus-32'
# MEAname = 'Neuronexus-32-cut-30'
cell_sel = cell_name
# cell_sel2 = cell_names[10]
print cell_sel

print 'Creating MEA'
mea_pos, mea_dim, mea_pitch = mea.return_site_positions(MEAname)

# instantiate cell
morphology_file = [join(cell_folder, 'morphology',f)
                   for f in os.listdir(join(cell_folder, 'morphology')) if 'asc' in f][0]
cell = LFPy.Cell(morphology=morphology_file, pt3d=True)
cell.set_rotation(np.pi/2., 0, np.pi/2.)

fig1 = plt.figure(figsize=(7,14))
ax1 = fig1.add_subplot(1, 1, 1, projection='3d')

verts = []
elec_size = 7.5
lim = [-200, 200]
ax1.view_init(elev=10, azim=45)

rot_pos = plot_probe_3d(mea_pos, rot_axis=[0,1,0],theta=-np.pi/4., ax=ax1, probe_name='neuronexus', alpha=0.7)
                        #xlim=lim, ylim=lim, zlim=lim)
#from Neto
min_dist = 48
#closest-electrode 29 -> 0
closest = rot_pos[0]
normal = np.abs(np.cross(rot_pos[1]-rot_pos[0], rot_pos[-1]-rot_pos[0]))
normal /= np.linalg.norm(normal)
new_pos = rot_pos[0] + min_dist*normal
plot_neuron(cell=cell, plane='3d', ax=ax1, color=[0.3, 0.3, 0.3], alpha=0.7, c_soma='g', pos=new_pos) #, condition='cell.xend[idx] > 20')

#plot pipette
pipette_dir = [0.5, 0, np.sqrt(3)/2.]
rad = 5
length = 1000
bottom = cell.somapos + [cell.diam[0]/2., 0, 0]

plot_cylinder_3d(bottom, pipette_dir, length, rad, ax=ax1, alpha=.2, color=[0.3, 0.7, 0.7])

lim=200
# Get rid of the panes
ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax1.w_zaxis.set_pane_color((1.0, 1.0, 1., 0.0))

# Get rid of the spines
ax1.w_xaxis.line.set_color((1.0, 1.0, 1., 0.0))
ax1.w_yaxis.line.set_color((1.0, 1.0, 1., 0.0))
ax1.w_zaxis.line.set_color((1.0, 1.0, 1., 0.0))

lim=300
ax1.set_xlim3d(-lim, lim)
ax1.set_ylim3d(-lim, lim)
ax1.set_zlim3d(-lim, 2*lim)

# ax1.set_xticklabels([])
# ax1.set_yticklabels([])
# ax1.set_zticklabels([])
ax1.axis('off')
plt.tight_layout()

plt.ion()
plt.show()
#
#
if save_fig:
    plt.savefig('../figures/neto.pdf')
    plt.savefig('../figures/neto.svg')
    plt.savefig('../figures/neto.png')

