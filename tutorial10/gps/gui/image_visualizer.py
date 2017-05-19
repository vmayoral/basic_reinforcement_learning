"""
Image Visualizer

The Image Visualizer class expects to be given images in real time (via the
update function), and displays those images onto the screen. Optionally,
images can be supplied via a rostopic.

The Image Visualizer can also overlay an initial image or target image on top 
of the realtime images. This is used for resetting the task space after each
sample if the robot disturbed it during sampling.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gps.gui.action_panel import Action, ActionPanel

import logging
LOGGER = logging.getLogger(__name__)
ROS_ENABLED = False
try:
    import rospkg
    import roslib
    import rospy
    from sensor_msgs.msg import Image

    roslib.load_manifest('gps_agent_pkg')
    ROS_ENABLED = True
except ImportError as e:
    LOGGER.debug('Import ROS failed: %s', e)
except rospkg.common.ResourceNotFound as e:
    LOGGER.debug('No gps_agent_pkg: %s', e)


class ImageVisualizer(object):

    def __init__(self, fig, gs, cropsize=None, rostopic=None, show_overlay_buttons=False):
        """
        If rostopic is given to this constructor, then the image visualizer will 
        automatically update with rostopic image. Else, the update method must 
        be manually called to supply images.
        """
        # Real-time image
        self._t = 0
        self._current_image = None
        self._crop_size = cropsize

        # Image overlay
        self._initial_image_overlay_on = False
        self._target_image_overlay_on = False
        self._initial_image = None
        self._initial_alpha = None
        self._target_image = None
        self._target_alpha = None
        self._default_image = np.zeros((1, 1, 3))
        self._default_alpha = 0.0

        # Actions
        actions_arr = [
            Action('oii', 'overlay_initial_image', self.toggle_initial_image_overlay, axis_pos=0),
            Action('oti', 'overlay_target_image',  self.toggle_target_image_overlay,  axis_pos=1),
        ]

        # GUI Components
        self._fig = fig
        self._gs = gridspec.GridSpecFromSubplotSpec(8, 1, subplot_spec=gs)
        self._gs_action_panel = self._gs[0:1, 0]
        self._gs_image_axis  = self._gs[1:8, 0]

        if show_overlay_buttons:
            self._action_panel = ActionPanel(self._fig, self._gs_action_panel, 1, 2, actions_arr)

        self._ax_image = plt.subplot(self._gs_image_axis)
        self._ax_image.set_axis_off()
        self._plot = self._ax_image.imshow(self._default_image)
        self._overlay_plot_initial = self._ax_image.imshow(self._default_image, alpha=self._default_alpha)
        self._overlay_plot_target  = self._ax_image.imshow(self._default_image, alpha=self._default_alpha)

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()   # Fixes bug with Qt4Agg backend

        # ROS subscriber for PS3 controller
        if rostopic and ROS_ENABLED:
            rospy.Subscriber(rostopic, Image, self.update_ros, queue_size=1,
                             buff_size=2**24)

    def update(self, image):
        """ Update images. """
        if image is None:
            return
        image = np.array(image, dtype=float)
        if self._crop_size:
            h, w = image.shape[0], image.shape[1]
            ch, cw = self._crop_size[0], self._crop_size[1]
            image = image[(h/2-ch/2):(h/2-ch/2+ch), (w/2-cw/2):(w/2-cw/2+cw), :]

        self._current_image = image
        self._plot.set_array(image)
        self.draw()

    def update_ros(self, image_msg):
        # Extract image.
        image = np.fromstring(image_msg.data, np.uint8)
        # Convert from ros image format to matplotlib image format.
        image = image.reshape(image_msg.height, image_msg.width, 3)[:, :, ::-1]
        image = 255 - image
        # Update visualizer.
        self.update(image)

    def get_current_image(self):
        return self._current_image

    def set_initial_image(self, image, alpha=0.3):
        if image is None:
            return
        self._initial_image = np.array(image, dtype=float)
        self._initial_alpha = alpha

    def set_target_image(self, image, alpha=0.3):
        if image is None:
            return
        self._target_image = np.array(image, dtype=float)
        self._target_alpha = alpha

    def toggle_initial_image_overlay(self, event=None):
        self._initial_image_overlay_on = not self._initial_image_overlay_on
        if self._initial_image is not None and self._initial_image_overlay_on:
            image = self._initial_image
        else:
            image = self._default_image
        if self._initial_alpha is not None and self._initial_image_overlay_on:
            alpha = self._initial_alpha
        else:
            alpha = self._default_alpha
        self._overlay_plot_initial.set_array(image)
        self._overlay_plot_initial.set_alpha(alpha)
        self.draw()

    def toggle_target_image_overlay(self, event=None):
        self._target_image_overlay_on = not self._target_image_overlay_on
        if self._target_image is not None and self._target_image_overlay_on:
            image = self._target_image
        else:
            image = self._default_image
        if self._target_alpha is not None and self._target_image_overlay_on:
            alpha = self._target_alpha
        else:
            alpha = self._default_alpha
        self._overlay_plot_target.set_array(image)
        self._overlay_plot_target.set_alpha(alpha)
        self.draw()

    def draw(self):
        self._ax_image.draw_artist(self._ax_image.patch)
        self._ax_image.draw_artist(self._plot)
        self._ax_image.draw_artist(self._overlay_plot_initial)
        self._ax_image.draw_artist(self._overlay_plot_target)
        self._fig.canvas.update()
        self._fig.canvas.flush_events()   # Fixes bug with Qt4Agg backend
