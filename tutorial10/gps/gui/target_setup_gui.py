"""
Target Setup GUI

The Target Setup GUI is used to interact with the GPS algorithm before training.
It contains the below four functionalities:

Action Panel                contains buttons for:
                            - switching target numbers,
                            - switching actuator types
                            - setting initial/target positions
                            - setting initial/target images
                            - moving to initial/target positions
Action Status Textbox       displays action status
Targets Textbox             displays targets values for current target/actuator
Image Visualizer            displays images received from a rostopic

For more detailed documentation, visit: rll.berkeley.edu/gps/gui
"""
import os
import subprocess
import signal
DEVNULL = open(os.devnull, 'wb')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gps.gui.config import config
from gps.gui.action_panel import Action, ActionPanel
from gps.gui.textbox import Textbox
from gps.gui.image_visualizer import ImageVisualizer
from gps.gui.util import load_pose_from_npz, load_data_from_npz, \
        save_pose_to_npz, save_data_to_npz

from gps.proto.gps_pb2 import END_EFFECTOR_POSITIONS, END_EFFECTOR_ROTATIONS, \
        JOINT_ANGLES, JOINT_SPACE

import logging
LOGGER = logging.getLogger(__name__)
ROS_ENABLED = False
try:
    import rospkg
    from gps.agent.ros.ros_utils import TimeoutException

    ROS_ENABLED = True
except ImportError as e:
    LOGGER.debug('Import ROS failed: %s', e)
except rospkg.common.ResourceNotFound as e:
    LOGGER.debug('No gps_agent_pkg: %s', e)


class TargetSetupGUI(object):

    def __init__(self, hyperparams, agent):
        self._hyperparams = hyperparams
        self._agent = agent

        self._log_filename = self._hyperparams['log_filename']
        self._target_filename = self._hyperparams['target_filename']

        self._num_targets = config['num_targets']
        self._actuator_types = config['actuator_types']
        self._actuator_names = config['actuator_names']
        self._num_actuators = len(self._actuator_types)

        # Target Setup Status.
        self._target_number = 0
        self._actuator_number = 0
        self._actuator_type = self._actuator_types[self._actuator_number]
        self._actuator_name = self._actuator_names[self._actuator_number]
        self._initial_position = ('unknown', 'unknown', 'unknown')
        self._target_position  = ('unknown', 'unknown', 'unknown')
        self._initial_image = None
        self._target_image  = None
        self._mannequin_mode = False
        self._mm_process = None

        # Actions.
        actions_arr = [
            Action('ptn', 'prev_target_number', self.prev_target_number, axis_pos=0),
            Action('ntn', 'next_target_number', self.next_target_number, axis_pos=1),
            Action('pat', 'prev_actuator_type', self.prev_actuator_type, axis_pos=2),
            Action('nat', 'next_actuator_type', self.next_actuator_type, axis_pos=3),

            Action('sip', 'set_initial_position', self.set_initial_position, axis_pos=4),
            Action('stp', 'set_target_position',  self.set_target_position,  axis_pos=5),
            Action('sii', 'set_initial_image',    self.set_initial_image,    axis_pos=6),
            Action('sti', 'set_target_image',     self.set_target_image,     axis_pos=7),

            Action('mti', 'move_to_initial',  self.move_to_initial,  axis_pos=8),
            Action('mtt', 'move_to_target',   self.move_to_target,   axis_pos=9),
            Action('rc',  'relax_controller', self.relax_controller, axis_pos=10),
            Action('mm',  'mannequin_mode',   self.mannequin_mode,   axis_pos=11),
        ]

        # Setup figure.
        plt.ion()
        plt.rcParams['toolbar'] = 'None'
        for key in plt.rcParams:
            if key.startswith('keymap.'):
                plt.rcParams[key] = ''

        self._fig = plt.figure(figsize=config['figsize'])
        self._fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99,
                wspace=0, hspace=0)

        # Assign GUI component locations.
        self._gs = gridspec.GridSpec(4, 4)
        self._gs_action_panel                   = self._gs[0:1, 0:4]
        if config['image_on']:
            self._gs_target_output              = self._gs[1:3, 0:2]
            self._gs_initial_image_visualizer   = self._gs[3:4, 0:1]
            self._gs_target_image_visualizer    = self._gs[3:4, 1:2]
            self._gs_action_output              = self._gs[1:2, 2:4]
            self._gs_image_visualizer           = self._gs[2:4, 2:4]
        else:
            self._gs_target_output              = self._gs[1:4, 0:2]
            self._gs_action_output              = self._gs[1:4, 2:4]

        # Create GUI components.
        self._action_panel = ActionPanel(self._fig, self._gs_action_panel, 3, 4, actions_arr)
        self._target_output = Textbox(self._fig, self._gs_target_output,
                log_filename=self._log_filename,
                fontsize=config['target_output_fontsize'])
        self._action_output = Textbox(self._fig, self._gs_action_output)
        if config['image_on']:
            self._initial_image_visualizer = ImageVisualizer(self._fig,
                    self._gs_initial_image_visualizer)
            self._target_image_visualizer = ImageVisualizer(self._fig,
                    self._gs_target_image_visualizer)
            self._image_visualizer = ImageVisualizer(self._fig, 
                    self._gs_image_visualizer, cropsize=config['image_size'],
                    rostopic=config['image_topic'], show_overlay_buttons=True)

        # Setup GUI components.
        self.reload_positions()
        self.update_target_text()
        self.set_action_text('Press an action to begin.')
        self.set_action_bgcolor('white')

        self._fig.canvas.draw()

    # Target Setup Functions.
    def prev_target_number(self, event=None):
        self.set_action_status_message('prev_target_number', 'requested')
        self._target_number = (self._target_number - 1) % self._num_targets
        self.reload_positions()
        self.update_target_text()
        self.set_action_text()
        self.set_action_status_message('prev_target_number', 'completed',
                message='target number = %d' % self._target_number)

    def next_target_number(self, event=None):
        self.set_action_status_message('next_target_number', 'requested')
        self._target_number = (self._target_number + 1) % self._num_targets
        self.reload_positions()
        self.update_target_text()
        self.set_action_text()
        self.set_action_status_message('next_target_number', 'completed',
                message='target number = %d' % self._target_number)

    def prev_actuator_type(self, event=None):
        self.set_action_status_message('prev_actuator_type', 'requested')
        self._actuator_number = (self._actuator_number-1) % self._num_actuators
        self._actuator_type = self._actuator_types[self._actuator_number]
        self._actuator_name = self._actuator_names[self._actuator_number]
        self.reload_positions()
        self.update_target_text()
        self.set_action_text()
        self.set_action_status_message('prev_actuator_type', 'completed',
                message='actuator name = %s' % self._actuator_name)

    def next_actuator_type(self, event=None):
        self.set_action_status_message('next_actuator_type', 'requested')
        self._actuator_number = (self._actuator_number+1) % self._num_actuators
        self._actuator_type = self._actuator_types[self._actuator_number]
        self._actuator_name = self._actuator_names[self._actuator_number]
        self.reload_positions()
        self.update_target_text()
        self.set_action_text()
        self.set_action_status_message('next_actuator_type', 'completed',
                message='actuator name = %s' % self._actuator_name)

    def set_initial_position(self, event=None):
        self.set_action_status_message('set_initial_position', 'requested')
        try:
            sample = self._agent.get_data(arm=self._actuator_type)
        except TimeoutException:
            self.set_action_status_message('set_initial_position', 'failed',
                    message='TimeoutException while retrieving sample')
            return
        ja = sample.get(JOINT_ANGLES)
        ee_pos = sample.get(END_EFFECTOR_POSITIONS)
        ee_rot = sample.get(END_EFFECTOR_ROTATIONS)
        self._initial_position = (ja, ee_pos, ee_rot)
        save_pose_to_npz(self._target_filename, self._actuator_name,
                str(self._target_number), 'initial', self._initial_position)

        self.update_target_text()
        self.set_action_status_message('set_initial_position', 'completed',
                message='initial position =\n %s' % \
                self.position_to_str(self._initial_position))

    def set_target_position(self, event=None):
        self.set_action_status_message('set_target_position', 'requested')
        try:
            sample = self._agent.get_data(arm=self._actuator_type)
        except TimeoutException:
            self.set_action_status_message('set_target_position', 'failed',
                    message='TimeoutException while retrieving sample')
            return
        ja = sample.get(JOINT_ANGLES)
        ee_pos = sample.get(END_EFFECTOR_POSITIONS)
        ee_rot = sample.get(END_EFFECTOR_ROTATIONS)
        self._target_position = (ja, ee_pos, ee_rot)
        save_pose_to_npz(self._target_filename, self._actuator_name,
                str(self._target_number), 'target', self._target_position)

        self.update_target_text()
        self.set_action_status_message('set_target_position', 'completed',
                message='target position =\n %s' % \
                self.position_to_str(self._target_position))

    def set_initial_image(self, event=None):
        self.set_action_status_message('set_initial_image', 'requested')
        self._initial_image = self._image_visualizer.get_current_image()
        if self._initial_image is None:
            self.set_action_status_message('set_initial_image', 'failed',
                    message='no image available')
            return
        save_data_to_npz(self._target_filename, self._actuator_name,
                str(self._target_number), 'initial', 'image', self._initial_image)

        self.update_target_text()
        self.set_action_status_message('set_initial_image', 'completed',
                message='initial image =\n %s' % str(self._initial_image))

    def set_target_image(self, event=None):
        self.set_action_status_message('set_target_image', 'requested')
        self._target_image = self._image_visualizer.get_current_image()
        if self._target_image is None:
            self.set_action_status_message('set_target_image', 'failed',
                    message='no image available')
            return
        save_data_to_npz(self._target_filename, self._actuator_name,
                str(self._target_number), 'target', 'image', self._target_image)

        self.update_target_text()
        self.set_action_status_message('set_target_image', 'completed',
                message='target image =\n %s' % str(self._target_image))

    def move_to_initial(self, event=None):
        ja = self._initial_position[0]
        self.set_action_status_message('move_to_initial', 'requested')
        self._agent.reset_arm(self._actuator_type, JOINT_SPACE, ja.T)
        self.set_action_status_message('move_to_initial', 'completed',
                message='initial position: %s' % str(ja))

    def move_to_target(self, event=None):
        ja = self._target_position[0]
        self.set_action_status_message('move_to_target', 'requested')
        self._agent.reset_arm(self._actuator_type, JOINT_SPACE, ja.T)
        self.set_action_status_message('move_to_target', 'completed',
                message='target position: %s' % str(ja))

    def relax_controller(self, event=None):
        self.set_action_status_message('relax_controller', 'requested')
        self._agent.relax_arm(self._actuator_type)
        self.set_action_status_message('relax_controller', 'completed',
                message='actuator name: %s' % self._actuator_name)

    def mannequin_mode(self, event=None):
        """
        Calls "roslaunch pr2_mannequin_mode pr2_mannequin_mode.launch"
        (only works for the PR2 robot).
        """
        if not self._mannequin_mode:
            self.set_action_status_message('mannequin_mode', 'requested')
            subprocess.Popen(['rosrun', 'pr2_controller_manager', 
                    'pr2_controller_manager', 'stop', 'GPSPR2Plugin'], stdout=DEVNULL)
            self._mm_process = subprocess.Popen(['roslaunch',
                    'pr2_mannequin_mode', 'pr2_mannequin_mode.launch'], stdout=DEVNULL)
            self._mannequin_mode = True
            self.set_action_status_message('mannequin_mode', 'completed',
                    message='mannequin mode toggled on')
        else:
            self.set_action_status_message('mannequin_mode', 'requested')
            self._mm_process.send_signal(signal.SIGINT)
            subprocess.Popen(['rosrun', 'pr2_controller_manager',
                    'pr2_controller_manager', 'start', 'GPSPR2Plugin'], stdout=DEVNULL)
            self._mannequin_mode = False
            self.set_action_status_message('mannequin_mode', 'completed',
                    message='mannequin mode toggled off')

    # GUI functions.
    def update_target_text(self):
        np.set_printoptions(precision=3, suppress=True)
        text = (
            'target number = %s\n' % str(self._target_number) +
            'actuator name = %s\n' % str(self._actuator_name) +
            '\ninitial position\n%s' % self.position_to_str(self._initial_position) +
            '\ntarget position\n%s' % self.position_to_str(self._target_position) +
            '\ninitial image (left) =\n%s\n' % str(self._initial_image) +
            '\ntarget image (right) =\n%s\n' % str(self._target_image)
        )
        self._target_output.set_text(text)

        if config['image_on']:
            self._initial_image_visualizer.update(self._initial_image)
            self._target_image_visualizer.update(self._target_image)
            self._image_visualizer.set_initial_image(self._initial_image, alpha=0.3)
            self._image_visualizer.set_target_image(self._target_image, alpha=0.3)

    def position_to_str(self, position):
        np.set_printoptions(precision=3, suppress=True)
        ja, ee_pos, ee_rot = position
        return ('joint angles =\n%s\n'           % ja +
                'end effector positions =\n%s\n' % ee_pos +
                'end effector rotations =\n%s\n' % ee_rot)

    def set_action_status_message(self, action, status, message=None):
        text = action + ': ' + status
        if message:
            text += '\n\n' + message
        self.set_action_text(text)
        if status == 'requested':
            self.set_action_bgcolor('yellow')
        elif status == 'completed':
            self.set_action_bgcolor('green')
        elif status == 'failed':
            self.set_action_bgcolor('red')

    def set_action_text(self, text=''):
        self._action_output.set_text(text)

    def set_action_bgcolor(self, color, alpha=1.0):
        self._action_output.set_bgcolor(color, alpha)

    def reload_positions(self):
        """
        Reloads the initial/target positions and images.
        This is called after the target number of actuator type have changed.
        """
        self._initial_position = load_pose_from_npz(self._target_filename,
                self._actuator_name, str(self._target_number), 'initial')
        self._target_position  = load_pose_from_npz(self._target_filename,
                self._actuator_name, str(self._target_number), 'target')
        self._initial_image    = load_data_from_npz(self._target_filename,
                self._actuator_name, str(self._target_number), 'initial',
                'image', default=None)
        self._target_image     = load_data_from_npz(self._target_filename,
                self._actuator_name, str(self._target_number), 'target',
                'image', default=None)
