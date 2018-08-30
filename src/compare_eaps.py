import numpy as np
import matplotlib.pylab as plt
import yaml

try:
    from neuroplot import *
    neur_plot=True
except ImportError:
    neur_plot=False
from os.path import join
import sys


if __name__ == '__main__':

    if '-f' in sys.argv:
        pos = sys.argv.index('-f')
        mesh = sys.argv[pos + 1]
    else:
        raise Exception('Indicate fem results with -f argument')

    if 'noprobe' in mesh:
        no_mesh = mesh
        w_mesh = mesh[:no_mesh.find('noprobe')] + 'wprobe'
    elif 'wprobe' in mesh:
        w_mesh = mesh
        no_mesh = mesh[:w_mesh.find('wprobe')] + 'noprobe'

    conv = 1E-4
    colors = plt.rcParams['axes.color_cycle']
    fs_legend = 20
    save_fig = False
    figsize = (9, 14)

    if 'fancy' in no_mesh:
        probe = 'MEA'
        pitch = np.array([18., 25.])
    elif 'pixel' in no_mesh:
        probe = 'pixel'
        pitch = np.array([12., 12.])
    else:
        probe = 'microwire'



    with open(join(no_mesh, 'params.yaml'), 'r') as f:
        info = yaml.load(f)

    T = info['problem']['Tstop']

    times = np.load(join(no_mesh, 'times.npy'))
    sites = np.load(join(no_mesh, 'sites.npy')) / conv
    dist = np.array([np.linalg.norm(p) for p in sites])

    v_noprobe = np.load(join(no_mesh, 'v_probe.npy'))*1000
    v_wprobe = np.load(join(w_mesh, 'v_probe.npy'))*1000

    v_p = np.squeeze(np.array([v_noprobe, v_wprobe]))


    if neur_plot and len(v_p.shape) == 3:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1)
        ax = plot_mea_recording(v_p, sites, pitch, ax=ax, time=T, lw=2, scalebar=True, vscale=40)
        ax.legend(labels=['Without probe', 'With probe'], fontsize=fs_legend, loc='upper right')
        fig.tight_layout()
    else:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(times, np.transpose(v_p), lw=2)
        ax.legend(labels=['Without probe', 'With probe'], fontsize=fs_legend)
        ax.axis('off')
        shift = 0.01*times[-1]
        pos_h = [np.min(times), np.min(v_p)]
        length_h = (np.max(v_p) - np.min(v_p))/5.
        pos_w = [np.min(times), np.min(v_p)]
        length_w = times[-1] / 5

        ax.plot([pos_h[0], pos_h[0]], [pos_h[1], pos_h[1] + length_h], color='k', lw=2)
        ax.text(pos_h[0] + shift, pos_h[1] + length_h / 2., str(int((np.max(v_p) - np.min(v_p)) // 10 * 10)) + ' $\mu$V')
        ax.plot([pos_w[0], pos_w[0] + length_w], [pos_w[1], pos_w[1]], color='k', lw=2)
        ax.text(pos_w[0] + shift, pos_w[1] - length_h / 4., str(round(times[-1] / 5, 1)) + ' ms')

    print 'NO PROBE: ', np.min(v_noprobe)
    print 'WITH PROBE: ', np.min(v_wprobe)
    print 'Average: ', np.mean(np.abs(v_wprobe - v_noprobe)), ' +- ', np.std(np.abs(v_wprobe - v_noprobe))
    abs_diff = np.abs(v_wprobe - v_noprobe)
    v_max_w = np.max(np.abs(v_wprobe), axis=1)
    v_max_no = np.max(np.abs(v_noprobe), axis=1)
    ratios = v_max_w / v_max_no


    print 'DIFF: ', np.max(np.abs(v_wprobe - v_noprobe)) #np.min(v_noprobe) - np.min(v_wprobe)

    print 'Min occurrence with probe: ', np.unravel_index(v_wprobe.argmin(), v_wprobe.shape)
    print 'Min occurrence no probe: ', np.unravel_index(v_noprobe.argmin(), v_noprobe.shape)

    ratio = np.min(v_wprobe)/np.min(v_noprobe)
    v_corr = ratio * v_noprobe
    v_p_corr = np.squeeze(np.array([v_wprobe, v_corr]))

    print 'Average: ', np.mean(np.abs(v_wprobe - v_corr)), ' +- ', np.std(np.abs(v_wprobe - v_corr))
    abs_diff_corr = np.abs(v_wprobe - v_corr)
    print 'DIFFERENCE AFTER CORRECTION: ', np.max(np.abs(v_wprobe - v_corr))
    print 'Max difference after correction', np.unravel_index(np.argmax(np.abs(v_wprobe - v_corr)), (v_wprobe - v_corr).shape)

    if neur_plot and len(v_p.shape) == 3:
        fig_c = plt.figure(figsize=figsize)
        ax = fig_c.add_subplot(1,1,1)
        ax = plot_mea_recording(v_p_corr, sites, pitch, ax=ax, time=T, lw=2, colors=colors[1:3], scalebar=True, vscale=40)
        ax.legend(labels=['With probe', 'Corrected'], fontsize=fs_legend, loc='upper right')
        fig_c.tight_layout()
    else:
        fig_c = plt.figure(figsize=figsize)
        ax = fig_c.add_subplot(1, 1, 1)
        ax.plot(times, v_p_corr[0], lw=2, color=colors[1])
        ax.plot(times, v_p_corr[1], lw=2, color=colors[2])
        ax.legend(labels=['With probe', 'Corrected'], fontsize=fs_legend)
        ax.axis('off')
        shift = 0.01*times[-1]
        pos_h = [np.min(times), np.min(v_p)]
        length_h = (np.max(v_p) - np.min(v_p))/5.
        pos_w = [np.min(times), np.min(v_p)]
        length_w = times[-1] / 5

        ax.plot([pos_h[0], pos_h[0]], [pos_h[1], pos_h[1] + length_h], color='k', lw=2)
        ax.text(pos_h[0] + shift, pos_h[1] + length_h / 2., str(int((np.max(v_p) - np.min(v_p)) // 10 * 10)) + ' $\mu$V')
        ax.plot([pos_w[0], pos_w[0] + length_w], [pos_w[1], pos_w[1]], color='k', lw=2)
        ax.text(pos_w[0] + shift, pos_w[1] - length_h / 4., str(round(times[-1] / 5, 1)) + ' ms')

    if save_fig:
        fig.savefig(join('figures', probe + '_EAP.pdf'))
        fig_c.savefig(join('figures', probe + '_EAP_corr.pdf'))

    plt.ion()
    plt.show()
