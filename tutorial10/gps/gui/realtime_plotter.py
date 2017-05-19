"""
Realtime Plotter

The Realtime Plotter expects to be constantly given values to plot in realtime.
It assumes the values are an array and plots different indices at different
colors according to the spectral colormap.
"""
import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec

from gps.gui.util import buffered_axis_limits


class RealtimePlotter(object):

    def __init__(self, fig, gs, time_window=500, labels=None, alphas=None):
        self._fig = fig
        self._gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs)
        self._ax = plt.subplot(self._gs[0])

        self._time_window = time_window
        self._labels = labels
        self._alphas = alphas
        self._init = False

        if self._labels:
            self.init(len(self._labels))

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()   # Fixes bug with Qt4Agg backend

    def init(self, data_len):
        """
        Initialize plots based off the length of the data array.
        """
        self._t = 0
        self._data_len = data_len
        self._data = np.empty((0, data_len))

        cm = plt.get_cmap('spectral')
        self._plots = []
        for i in range(data_len):
            color = cm(1.0 * i / data_len)
            alpha = self._alphas[i] if self._alphas is not None else 1.0
            label = self._labels[i] if self._labels is not None else str(i)
            self._plots.append(
                self._ax.plot([], [], color=color, alpha=alpha, label=label)[0]
            )
        self._ax.set_xlim(0, self._time_window)
        self._ax.set_ylim(0, 1)
        self._ax.legend(loc='upper left', bbox_to_anchor=(0, 1.15))

        self._init = True

    def update(self, x):
        """
        Update the plots with new data x. Assumes x is a one-dimensional array.
        """
        x = np.ravel([x])

        if not self._init:
            self.init(x.shape[0])

        assert x.shape[0] == self._data_len
        x = x.reshape((1, self._data_len))

        self._t += 1
        self._data = np.append(self._data, x, axis=0)

        t, tw = self._t, self._time_window
        t0, tf = (0, t) if t < tw else (t - tw, t)
        for i in range(self._data_len):
            self._plots[i].set_data(np.arange(t0, tf), self._data[t0:tf, i])

        x_range = (0, tw) if t < tw else (t - tw, t)
        self._ax.set_xlim(x_range)

        y_min, y_max = np.amin(self._data[t0:tf, :]), np.amax(self._data[t0:tf, :])
        self._ax.set_ylim(buffered_axis_limits(y_min, y_max, buffer_factor=1.25))

        self.draw()

    def draw(self):
        self._ax.draw_artist(self._ax.patch)
        for plot in self._plots:
            self._ax.draw_artist(plot)
        self._fig.canvas.update()
        self._fig.canvas.flush_events()   # Fixes bug with Qt4Agg backend
