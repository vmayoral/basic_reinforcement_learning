"""
Plotter 3D

The Plotter 3D plots data in 3D. It has options for setting a title and legend,
plotting 3D points or 3D Gaussians, and clipping data based off axis limits.

This is used to plot the 3D trajectories, including the trajectory samples,
policy samples, and the linear Gaussian controllers.
"""
import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

class Plotter3D:
    def __init__(self, fig, gs, num_plots, rows=None, cols=None):
        if cols is None:
            cols = int(np.floor(np.sqrt(num_plots)))
        if rows is None:
            rows = int(np.ceil(float(num_plots)/cols))
        assert num_plots <= rows*cols, 'Too many plots to put into gridspec.'

        self._fig = fig
        self._gs = gridspec.GridSpecFromSubplotSpec(8, 1, subplot_spec=gs)
        self._gs_legend = self._gs[0:1, 0]
        self._gs_plot   = self._gs[1:8, 0]

        self._ax_legend = plt.subplot(self._gs_legend)
        self._ax_legend.get_xaxis().set_visible(False)
        self._ax_legend.get_yaxis().set_visible(False)

        self._gs_plots = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=self._gs_plot)
        self._axarr = [plt.subplot(self._gs_plots[i], projection='3d') for i in range(num_plots)]
        self._lims = [None for i in range(num_plots)]
        self._plots = [[] for i in range(num_plots)]

        for ax in self._axarr:
            ax.tick_params(pad=0)
            ax.locator_params(nbins=5)
            for item in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
                item.set_fontsize(10)

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()   # Fixes bug with Qt4Agg backend

    def set_title(self, i, title):
        self._axarr[i].set_title(title)
        self._axarr[i].title.set_fontsize(10)

    def add_legend(self, linestyle, marker, color, label):
        self._ax_legend.plot([], [], linestyle=linestyle, marker=marker,
                color=color, label=label)
        self._ax_legend.legend(ncol=2, mode='expand', fontsize=10)

    def plot(self, i, xs, ys, zs, linestyle='-', linewidth=1.0, marker=None,
            markersize=5.0, markeredgewidth=1.0, color='black', alpha=1.0, label=''):
        # Manually clip at xlim, ylim, zlim (MPL doesn't support axis limits for 3D plots)
        if self._lims[i]:
            xlim, ylim, zlim = self._lims[i]
            xs[np.any(np.c_[xs < xlim[0], xs > xlim[1]], axis=1)] = np.nan
            ys[np.any(np.c_[ys < ylim[0], ys > ylim[1]], axis=1)] = np.nan
            zs[np.any(np.c_[zs < zlim[0], zs > zlim[1]], axis=1)] = np.nan

        # Create and add plot
        plot = self._axarr[i].plot(xs, ys, zs=zs, linestyle=linestyle,
                linewidth=linewidth, marker=marker, markersize=markersize,
                markeredgewidth=markeredgewidth, color=color, alpha=alpha,
                label=label)[0]
        self._plots[i].append(plot)

    def plot_3d_points(self, i, points, linestyle='-', linewidth=1.0,
            marker=None, markersize=5.0, markeredgewidth=1.0, color='black',
            alpha=1.0, label=''):
        self.plot(i, points[:, 0], points[:, 1], points[:, 2],
                linestyle=linestyle, linewidth=linewidth, marker=marker,
                markersize=markersize, markeredgewidth=markeredgewidth,
                color=color, alpha=alpha, label=label)

    def plot_3d_gaussian(self, i, mu, sigma, edges=100, linestyle='-.',
            linewidth=1.0, color='black', alpha=0.1, label=''):
        """
        Plots ellipses in the xy plane representing the Gaussian distributions
        specified by mu and sigma.
        Args:
            mu    - Tx3 mean vector for (x, y, z)
            sigma - Tx3x3 covariance matrix for (x, y, z)
            edges - the number of edges to use to construct each ellipse
        """
        p = np.linspace(0, 2*np.pi, edges)
        xy_ellipse = np.c_[np.cos(p), np.sin(p)]
        T = mu.shape[0]

        sigma_xy = sigma[:, 0:2, 0:2]
        u, s, v = np.linalg.svd(sigma_xy)

        for t in range(T):
            xyz = np.repeat(mu[t, :].reshape((1, 3)), edges, axis=0)
            xyz[:, 0:2] += np.dot(xy_ellipse, np.dot(np.diag(
                    np.sqrt(s[t, :])), u[t, :, :].T))
            self.plot_3d_points(i, xyz, linestyle=linestyle,
                    linewidth=linewidth, color=color, alpha=alpha, label=label)

    def set_lim(self, i, xlim, ylim, zlim):
        """
        Sets the xlim, ylim, and zlim for plot i
        WARNING: limits must be set before adding data to plots
        Args:
            xlim - a tuple of (x_start, x_end)
            ylim - a tuple of (y_start, y_end)
            zlim - a tuple of (z_start, z_end)
        """
        self._lims[i] = [xlim, ylim, zlim]

    def clear(self, i):
        for plot in self._plots[i]:
            plot.remove()
        self._plots[i] = []

    def clear_all(self):
        for i in range(len(self._plots)):
            self.clear(i)

    def draw(self):
        for ax in self._axarr:
            ax.draw_artist(ax.patch)
        for i in range(len(self._plots)):
            for plot in self._plots[i]:
                self._axarr[i].draw_artist(plot)
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()   # Fixes bug with Qt4Agg backend
