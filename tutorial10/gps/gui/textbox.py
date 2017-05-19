"""
Textbox

A Textbox represents the standard textbox. It has basic capabilities for
setting the text, appending text, or changing the background color.
If a log filename is given, all text displayed by the Textbox is also placed
within the log file.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ColorConverter


class Textbox:

    def __init__(self, fig, gs, log_filename=None, max_display_size=10,
        border_on=False, bgcolor=mpl.rcParams['figure.facecolor'], bgalpha=1.0,
        fontsize=12, font_family='sans-serif'):
        self._fig = fig
        self._gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs)
        self._ax = plt.subplot(self._gs[0])
        self._log_filename = log_filename

        self._text_box = self._ax.text(0.01, 0.95, '', color='black',
                va='top', ha='left', transform=self._ax.transAxes,
                fontsize=fontsize, family=font_family)
        self._text_arr = []
        self._max_display_size = max_display_size

        self._ax.set_xticks([])
        self._ax.set_yticks([])
        if not border_on:
            self._ax.spines['top'].set_visible(False)
            self._ax.spines['right'].set_visible(False)
            self._ax.spines['bottom'].set_visible(False)
            self._ax.spines['left'].set_visible(False)

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()     # Fixes bug with Qt4Agg backend
        self.set_bgcolor(bgcolor, bgalpha)  # this must come after fig.canvas.draw()

    #TODO: Add docstrings here.
    def set_text(self, text):
        self._text_arr = [text]
        self._text_box.set_text('\n'.join(self._text_arr))
        self.log_text(text)
        self.draw()

    def append_text(self, text):
        self._text_arr.append(text)
        if len(self._text_arr) > self._max_display_size:
            self._text_arr = self._text_arr[-self._max_display_size:]
        self._text_box.set_text('\n'.join(self._text_arr))
        self.log_text(text)
        self.draw()

    def log_text(self, text):
        if self._log_filename is not None:
            with open(self._log_filename, 'a') as f:
                f.write(text + '\n')

    def set_bgcolor(self, color, alpha=1.0):
        self._ax.set_axis_bgcolor(ColorConverter().to_rgba(color, alpha))
        self.draw()

    def draw(self):
        color, alpha = self._ax.get_axis_bgcolor(), self._ax.get_alpha()
        self._ax.set_axis_bgcolor(mpl.rcParams['figure.facecolor'])
        self._ax.draw_artist(self._ax.patch)
        self._ax.set_axis_bgcolor(ColorConverter().to_rgba(color, alpha))

        self._ax.draw_artist(self._ax.patch)
        self._ax.draw_artist(self._text_box)
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()   # Fixes bug with Qt4Agg backend
