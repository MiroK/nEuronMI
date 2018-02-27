'''
- load data from results folder
- plot minimum peak with probe, without probe, and difference depending on:
    - boxsize (1-2-3)
    - meshsize (0-1-2-3)
    - probe
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import os
from os.path import join
from neuroplot import *


fs_label = 18
fs_legend = 15
fs_title = 22
fs_ticks = 20
lw = 3
ms = 10

save_fig=False

data = pd.read_pickle(join('results', 'results.pkl'))

# #remove dist 2.5
# data = data[data['tip_x']!='20.0']
# data = data[data['tip_x']!='27.5']

data_40 = data[data['tip_x']=='40']
data_dist = data[data['tip_x']!='40']


###########################################
# CONVERGENCE
xlim = [10, 90]
ylim = [-130, 80]

data_fancy = data_dist[data_dist['probe']=='fancy']
data_cylinder = data_dist[data_dist['probe']=='cylinder']

fig1 = plt.figure(figsize=(15,7))
ax11 = fig1.add_subplot(1,3,1)
ax11.plot(data_fancy.tip_x.astype('float'), data_fancy.min_noprobe*1e3, linestyle='-', marker='^',
              label='peak without probe', lw=lw, ms=ms)
ax11.plot(data_fancy.tip_x.astype('float'), data_fancy.min_wprobe*1e3, linestyle='-', marker='o',
              label='peak with probe', lw=lw, ms=ms)
ax11.plot(data_fancy.tip_x.astype('float'), data_fancy['diff']*1e3, linestyle='-', marker='d',
              label='peak difference', lw=lw, ms=ms)
ax11.set_xlabel('Distance ($\mu$m)', fontsize = fs_label)
ax11.set_ylabel('V ($\mu$V)', fontsize = fs_label)
ax11.set_title('MEA probe', fontsize = fs_title)
ax11.legend(fontsize = fs_legend)
ax11.set_xlim(xlim)
ax11.set_ylim(ylim)


ax12 = fig1.add_subplot(1,3,2)
ax12.plot(data_cylinder.tip_x.astype('float'), data_cylinder.min_noprobe*1e3, linestyle='-', marker='^',
              label='peak without probe', lw=lw, ms=ms)
ax12.plot(data_cylinder.tip_x.astype('float'), data_cylinder.min_wprobe*1e3, linestyle='-', marker='o',
              label='peak with probe', lw=lw, ms=ms)
ax12.plot(data_cylinder.tip_x.astype('float'), data_cylinder['diff']*1e3, linestyle='-', marker='d',
              label='peak difference', lw=lw, ms=ms)
ax12.set_xlabel('Distance ($\mu$m)', fontsize = fs_label)
ax12.set_ylabel('V ($\mu$V)', fontsize = fs_label)
ax12.set_title('Microwire probe', fontsize = fs_title)
ax12.legend(fontsize = fs_legend)
ax12.set_xlim(xlim)
ax12.set_ylim(ylim)


ax13 = fig1.add_subplot(1,3,3)
ax13.plot(data_fancy.tip_x.astype('float'), data_fancy.min_wprobe/data_fancy.min_noprobe, marker='d',
              linestyle='-', lw=lw, label='MEA probe', color='r', ms=ms)
ax13.plot(data_cylinder.tip_x.astype('float'), data_cylinder.min_wprobe/data_cylinder.min_noprobe, marker='o',
              linestyle='-', lw=lw, label='Microwire', color='grey', ms=ms)
ax13.axhline(np.mean(data_fancy.min_wprobe/data_fancy.min_noprobe), color='r', alpha=0.4, ls='--')
ax13.axhline(np.mean(data_cylinder.min_wprobe/data_cylinder.min_noprobe), color='grey', alpha=0.4, ls='--')
ax13.text(87, np.mean(data_fancy.min_wprobe/data_fancy.min_noprobe)-0.1,
         str(round(np.mean(data_fancy.min_wprobe/data_fancy.min_noprobe),2)), color='r',
         alpha=0.5, fontsize=fs_label)
ax13.text(87, np.mean(data_cylinder.min_wprobe/data_cylinder.min_noprobe) - 0.1,
         str(round(np.mean(data_cylinder.min_wprobe/data_cylinder.min_noprobe),2)), color='grey',
         alpha=0.5, fontsize=fs_label)

ax13.set_xlabel('Distance ($\mu$m)', fontsize = fs_label)
ax13.set_ylabel('with / without probe', fontsize = fs_label)
ax13.legend(fontsize=fs_legend)
ax13.set_title('Peak ratio', fontsize=fs_title)

mark_subplots([ax11, ax12, ax13], xpos=-0.15, ypos=1, fs=40)
simplify_axes([ax11, ax12, ax13])
fig1.tight_layout()

###########################################
# CONVERGENCE

data_fancy_conv = data_40[data_40['probe']=='fancy']
data_cylinder_conv = data_40[data_40['probe']=='cylinder']

data_fancy_conv.min_noprobe = data_fancy_conv.min_noprobe*1000
data_fancy_conv.min_wprobe = data_fancy_conv.min_wprobe*1000

fig2 = plt.figure(figsize=(12, 7))
ax21 = fig2.add_subplot(1,2,1)
plt.xticks(ax21.get_xticks(), fontsize=fs_ticks)

ax22 = fig2.add_subplot(1,2,2)
plt.xticks(ax22.get_xticks(), fontsize=fs_ticks)

colors = plt.rcParams['axes.color_cycle']
labels = ['coarse 0', 'coarse 1', 'coarse 2', 'coarse 3']
sns.pointplot(x='box', y='min_wprobe', hue='coarse', data=data_fancy_conv, ax=ax21,
              palette=sns.color_palette('OrRd', 4), marker='o', label=labels)
sns.pointplot(x='box', y='min_noprobe', hue='coarse', data=data_fancy_conv, ax=ax22,
              palette=sns.color_palette('GnBu', 4), marker='^', label=labels)

print np.mean(data_fancy_conv.min_noprobe), ' +- ', np.std(data_fancy_conv.min_noprobe)
print np.mean(data_fancy_conv.min_wprobe), ' +- ', np.std(data_fancy_conv.min_wprobe)
print np.mean(data_fancy_conv['diff']*1000), ' +- ', np.std(data_fancy_conv['diff']*1000)

ax21.set_xlabel('Box size', fontsize = fs_label)
ax21.set_ylabel('V ($\mu$V)', fontsize = fs_label)
ax21.legend(fontsize=fs_legend)
leg_handles = ax21.get_legend_handles_labels()[0]
ax21.legend(leg_handles, labels, fontsize=fs_legend)
ax21.set_title('With MEA probe', fontsize=fs_title)

ax22.set_xlabel('Box size', fontsize = fs_label)
ax22.set_ylabel('V ($\mu$V)', fontsize = fs_label)
leg_handles = ax22.get_legend_handles_labels()[0]
ax22.legend(leg_handles, labels, fontsize=fs_legend)
ax22.set_title('Without probe', fontsize=fs_title)

mark_subplots([ax21, ax22], xpos=-0.1, ypos=1, fs=40)
simplify_axes([ax21, ax22])
fig2.tight_layout()


if save_fig:
    fig1.savefig('figures/distance_analysis.pdf')
    fig2.savefig('figures/convergence_analysis.pdf')

plt.ion()
plt.show()