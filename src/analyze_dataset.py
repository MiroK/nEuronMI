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

save_fig = True
plot_dist = False
plot_conv = False
plot_shift = True
plot_rot = True

data = pd.read_pickle(join('results', 'results.pkl'))

# #remove dist 2.5
# data = data[data['tip_x']!='20.0']
# data = data[data['tip_x']!='27.5']

data_40 = data[data['tip_x']=='40']
data_40 = data_40[data_40['tip_y']=='0']
data_dist = data[data['tip_x']!='40']
data_dist = data_dist[data_dist['tip_y']=='0']
data_dist = data_dist[data_dist['box']=='2']

figsize3 = (15,7)
figsize1 = (7,7)

colors = plt.rcParams['axes.color_cycle']


###########################################
# DISTANCE

xlim = [20, 100]
ylim = [-10, 100]

data_fancy_dist = data_dist[data_dist['probe']=='fancy'].sort_values(by=['tip_x'])
data_cylinder_dist = data_dist[data_dist['probe']=='cylinder'].sort_values(by=['tip_x'])
data_pixel_dist = data_dist[data_dist['probe']=='pixel'].sort_values(by=['tip_x'])

# data_fancy = data_fancy.sort_values(by=['tip_x'])
# data_cylinder = data_cylinder.sort_values(by=['tip_x'])

data_fancy_dist.min_noprobe = data_fancy_dist.min_noprobe*1000
data_fancy_dist.min_wprobe = data_fancy_dist.min_wprobe*1000
data_fancy_dist['diff'] = data_fancy_dist['diff']*1000
data_fancy_dist['ratios'] = np.round(data_fancy_dist.min_wprobe / data_fancy_dist.min_noprobe, 2)

data_cylinder_dist.min_noprobe = data_cylinder_dist.min_noprobe*1000
data_cylinder_dist.min_wprobe = data_cylinder_dist.min_wprobe*1000
data_cylinder_dist['diff'] = data_cylinder_dist['diff']*1000
data_cylinder_dist['ratios'] = np.round(data_cylinder_dist.min_wprobe / data_cylinder_dist.min_noprobe, 2)

data_pixel_dist.min_noprobe = data_pixel_dist.min_noprobe*1000
data_pixel_dist.min_wprobe = data_pixel_dist.min_wprobe*1000
data_pixel_dist['diff'] = data_pixel_dist['diff']*1000
data_pixel_dist['ratios'] = np.round(data_pixel_dist.min_wprobe / data_pixel_dist.min_noprobe, 2)

if plot_dist:

    fig11 = plt.figure(figsize=figsize1)
    ax111 = fig11.add_subplot(1,1,1)
    ax111.plot(data_fancy_dist.tip_x.astype('float'), np.abs(data_fancy_dist.min_noprobe), linestyle='-', marker='^',
                  label='peak without probe', lw=lw, ms=ms)
    ax111.plot(data_fancy_dist.tip_x.astype('float'), np.abs(data_fancy_dist.min_wprobe), linestyle='-', marker='o',
                  label='peak with probe', lw=lw, ms=ms)
    ax111.plot(data_fancy_dist.tip_x.astype('float'), data_fancy_dist['diff'], linestyle='-', marker='d',
                  label='peak difference', lw=lw, ms=ms)
    ax111.set_xlabel('Distance ($\mu$m)', fontsize = fs_label)
    ax111.set_ylabel('V ($\mu$V)', fontsize = fs_label)
    ax111.set_title('Neuronexus probe', fontsize = fs_title)
    ax111.legend(fontsize = fs_legend)
    ax111.set_xlim(xlim)
    ax111.set_ylim(ylim)


    fig12 = plt.figure(figsize=figsize1)
    ax112 = fig12.add_subplot(1,1,1)
    ax112.plot(data_cylinder_dist.tip_x.astype('float'), np.abs(data_cylinder_dist.min_noprobe), linestyle='-', marker='^',
                  label='peak without probe', lw=lw, ms=ms)
    ax112.plot(data_cylinder_dist.tip_x.astype('float'), np.abs(data_cylinder_dist.min_wprobe), linestyle='-', marker='o',
                  label='peak with probe', lw=lw, ms=ms)
    ax112.plot(data_cylinder_dist.tip_x.astype('float'), data_cylinder_dist['diff'], linestyle='-', marker='d',
                  label='peak difference', lw=lw, ms=ms)
    ax112.set_xlabel('Distance ($\mu$m)', fontsize = fs_label)
    ax112.set_ylabel('V ($\mu$V)', fontsize = fs_label)
    ax112.set_title('Microwire probe', fontsize = fs_title)
    ax112.legend(fontsize = fs_legend)
    ax112.set_xlim(xlim)
    ax112.set_ylim(ylim)

    fig13 = plt.figure(figsize=figsize1)
    ax113 = fig13.add_subplot(1,1,1)
    ax113.plot(data_pixel_dist.tip_x.astype('float'), np.abs(data_pixel_dist.min_noprobe), linestyle='-', marker='^',
                  label='peak without probe', lw=lw, ms=ms)
    ax113.plot(data_pixel_dist.tip_x.astype('float'), np.abs(data_pixel_dist.min_wprobe), linestyle='-', marker='o',
                  label='peak with probe', lw=lw, ms=ms)
    ax113.plot(data_pixel_dist.tip_x.astype('float'), data_pixel_dist['diff'], linestyle='-', marker='d',
                  label='peak difference', lw=lw, ms=ms)
    ax113.set_xlabel('Distance ($\mu$m)', fontsize = fs_label)
    ax113.set_ylabel('V ($\mu$V)', fontsize = fs_label)
    ax113.set_title('Neuropixel probe', fontsize = fs_title)
    ax113.legend(fontsize = fs_legend)
    ax113.set_xlim(xlim)
    ax113.set_ylim(ylim)


    fig14 = plt.figure(figsize=figsize1)
    ax114 = fig14.add_subplot(1,1,1)
    ax114.plot(data_fancy_dist.tip_x.astype('float'), data_fancy_dist.ratios, marker='d',
                  linestyle='-', lw=lw, label='Neuronexus probe', color='r', ms=ms)
    ax114.plot(data_pixel_dist.tip_x.astype('float'), data_pixel_dist.ratios, marker='d',
               linestyle='-', lw=lw, label='Neuropixel probe', color='b', ms=ms)
    ax114.plot(data_cylinder_dist.tip_x.astype('float'), data_cylinder_dist.ratios, marker='o',
                  linestyle='-', lw=lw, label='Microwire', color='grey', ms=ms)
    ax114.axhline(np.mean(data_fancy_dist.ratios), color='r', alpha=0.4, ls='--')
    ax114.axhline(np.mean(data_pixel_dist.ratios), color='b', alpha=0.4, ls='--')
    ax114.axhline(np.mean(data_cylinder_dist.ratios), color='grey', alpha=0.4, ls='--')
    ax114.text(87, np.mean(data_fancy_dist.ratios)-0.10,
             str(np.round(np.mean(data_fancy_dist.ratios), 2)), color='r',
             alpha=0.5, fontsize=fs_label)
    ax114.text(87, np.mean(data_pixel_dist.ratios) + 0.10,
               str(np.round(np.mean(data_pixel_dist.ratios), 2)), color='b',
               alpha=0.5, fontsize=fs_label)
    ax114.text(87, np.mean(data_cylinder_dist.ratios) - 0.10,
             str(np.round(np.mean(data_cylinder_dist.ratios), 2)), color='grey',
             alpha=0.5, fontsize=fs_label)

    ax114.set_xlabel('Distance ($\mu$m)', fontsize = fs_label)
    ax114.set_ylabel('ratio', fontsize = fs_label)
    ax114.legend(fontsize=fs_legend)
    ax114.set_title('Peak ratio', fontsize=fs_title)
    ax114.set_ylim([0.80, 2.5])



    simplify_axes([ax111, ax112, ax113, ax114])
    fig11.tight_layout()
    fig12.tight_layout()
    fig13.tight_layout()


###########################################
# CONVERGENCE

data_fancy_conv = data_40[data_40['probe']=='fancy']
data_cylinder_conv = data_40[data_40['probe']=='cylinder']

data_fancy_conv.min_noprobe = np.round(data_fancy_conv.min_noprobe*1000, 2)
data_fancy_conv.min_wprobe = np.round(data_fancy_conv.min_wprobe*1000, 2)
data_fancy_conv['diff'] = np.round(data_fancy_conv['diff']*1000, 2)
data_fancy_conv['ratios'] = np.round(data_fancy_conv.min_wprobe / data_fancy_conv.min_noprobe, 2)

data_fancy_conv = data_fancy_conv.sort_values(by=['box', 'coarse'])
print  data_fancy_conv.to_latex(columns=['coarse', 'box', 'min_wprobe', 'min_noprobe', 'diff', 'ratios'], index=False)

if plot_conv:
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
    print np.mean(data_fancy_conv['diff']), ' +- ', np.std(data_fancy_conv['diff'])

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

    fig22 = plt.figure()
    ax22 = fig22.add_subplot(1, 1, 1)
    im_val = np.array(data_fancy_conv.groupby(['box', 'coarse'])['ratios'].mean().unstack('box'))
    im = ax22.matshow(im_val)
    plt.colorbar(im)

###########################################
# SIDE SHIFT
data_fancy = data[data.probe=='fancy']
data_fancy = data_fancy[data_fancy.coarse.isin(['2'])]
data_fancy = data_fancy[data_fancy.box.isin(['5'])]
data_fancy['ratios'] = np.round(data_fancy.min_wprobe / data_fancy.min_noprobe, 2)
# data_fancy = data_fancy[data_fancy.box=='3']
data_fancy_shift = data_fancy[data_fancy.tip_y.isin(['0', '10.0', '20.0', '30.0', '40.0', '50.0', '60.0'])]
data_fancy_shift = data_fancy_shift[data_fancy_shift.tip_x.isin(['40', '40.0'])]
data_fancy_shift = data_fancy_shift.sort_values(by=['tip_y'])

if plot_shift:
    fig33 = plt.figure(figsize=figsize1)
    ax33 = fig33.add_subplot(1,1,1)
    # sns.pointplot(x="tip_y", y="ratios", data=data_fancy_shift, ax=ax33)
    ax33.plot(data_fancy_shift.tip_y.astype('float'), data_fancy_shift.ratios, marker='o', lw=2, color=colors[1])
    # sns.tsplot(data=data_fancy_shift.min_wprobe/data_fancy_shift.min_noprobe, err_style="ci_bars", color="r", ax=ax33)
    # ax33.plot(data_fancy_shift.tip_y.astype('float'), data_fancy_shift.min_wprobe/data_fancy_shift.min_noprobe, marker='d',
    #               linestyle='-', lw=lw, label='MEA probe', color='r', ms=ms)

    ax33.set_ylim([0.80, 2.00])
    ax33.set_xlabel('y_shift ($\mu$m)', fontsize = fs_label)
    ax33.set_ylabel('ratio', fontsize = fs_label)
    # ax33.legend(fontsize=fs_legend)
    ax33.set_title('Lateral shift', fontsize=fs_title)

    simplify_axes([ax33])
    fig33.tight_layout()


###########################################
# ROTATION
data_fancy = data[data.probe=='fancy']
# data_fancy = data_fancy[data_fancy.tip_x.isin(['70.0'])]
# data_fancy = data_fancy[data_fancy.box.isin(['4'])]
data_fancy['ratios'] = np.round(data_fancy.min_wprobe / data_fancy.min_noprobe, 2)

data_fancy_rot = data_fancy[data_fancy.tip_x.isin(['70.0'])]
data_fancy_rot = data_fancy_rot.sort_values(by=['rot'])

order = ['30', '60', '90', '120', '150', '180']

if plot_rot:
    fig44 = plt.figure(figsize=figsize1)
    ax44 = fig44.add_subplot(1,1,1)
    sns.pointplot(x="rot", y="ratios", data=data_fancy_rot, ax=ax44, order=order)
    # sns.tsplot(data=data_fancy_rot.min_wprobe/data_fancy_rot.min_noprobe, err_style="ci_bars", color="r", ax=ax33)
    # ax33.plot(data_fancy_rot.tip_y.astype('float'), data_fancy_rot.min_wprobe/data_fancy_rot.min_noprobe, marker='d',
    #               linestyle='-', lw=lw, label='MEA probe', color='r', ms=ms)

    # ax44.set_ylim([0.80, 2.00])
    ax44.set_xlabel('rotation ($\circ$)', fontsize = fs_label)
    ax44.set_ylabel('ratio ', fontsize = fs_label)
    ax44.legend(fontsize=fs_legend)
    ax44.set_title('Rotation', fontsize=fs_title)

    simplify_axes([ax44])
    fig44.tight_layout()


if save_fig:
    if plot_dist:
        fig11.savefig('figures/mea_distance.pdf')
        fig12.savefig('figures/microwire_distance.pdf')
        fig13.savefig('figures/pixel_distance.pdf')
        fig14.savefig('figures/ratio_distance.pdf')
    if plot_conv:
        fig2.savefig('figures/convergence_analysis.pdf')
    if plot_shift:
        fig33.savefig('figures/sideshift_all.pdf')
    if plot_rot:
        fig44.savefig('figures/rotations.pdf')

plt.ion()
plt.show()