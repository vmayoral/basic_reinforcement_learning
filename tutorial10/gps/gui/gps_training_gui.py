"""
GPS Training GUI

The GPS Training GUI is used to interact with the GPS algorithm during training.
It contains the below seven functionalities:

Action Panel                contains buttons for stop, reset, go, fail
Action Status Textbox       displays action status
Algorithm Status Textbox    displays algorithm status
Cost Plot                   displays costs after each iteration
Algorithm Output Textbox    displays algorithm output after each iteration
3D Trajectory Visualizer    displays 3D trajectories after each iteration
Image Visualizer            displays images received from a rostopic

For more detailed documentation, visit: rll.berkeley.edu/gps/gui
"""
import time
import threading

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gps.gui.config import config
from gps.gui.action_panel import Action, ActionPanel
from gps.gui.textbox import Textbox
from gps.gui.mean_plotter import MeanPlotter
from gps.gui.plotter_3d import Plotter3D
from gps.gui.image_visualizer import ImageVisualizer
from gps.gui.util import buffered_axis_limits, load_data_from_npz

from gps_pb2 import END_EFFECTOR_POINTS

# Needed for typechecks
#from gps.algorithm.algorithm_badmm import AlgorithmBADMM
#from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS

class GPSTrainingGUI(object):

    def __init__(self, hyperparams):
        self._hyperparams = hyperparams
        self._log_filename = self._hyperparams['log_filename']
        if 'target_filename' in self._hyperparams:
            self._target_filename = self._hyperparams['target_filename']
        else:
            self._target_filename = None

        # GPS Training Status.
        self.mode = config['initial_mode']  # Modes: run, wait, end, request, process.
        self.request = None                 # Requests: stop, reset, go, fail, None.
        self.err_msg = None
        self._colors = {
            'run': 'cyan',
            'wait': 'orange',
            'end': 'red',

            'stop': 'red',
            'reset': 'yellow',
            'go': 'green',
            'fail': 'magenta',
        }
        self._first_update = True

        # Actions.
        actions_arr = [
            Action('stop',  'stop',  self.request_stop,  axis_pos=0),
            Action('reset', 'reset', self.request_reset, axis_pos=1),
            Action('go',    'go',    self.request_go,    axis_pos=2),
            Action('fail',  'fail',  self.request_fail,  axis_pos=3),
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
        self._gs = gridspec.GridSpec(16, 8)
        self._gs_action_panel           = self._gs[0:2,  0:8]
        self._gs_action_output          = self._gs[2:3,  0:4]
        self._gs_status_output          = self._gs[3:4,  0:4]
        self._gs_cost_plotter           = self._gs[2:4,  4:8]
        self._gs_algthm_output          = self._gs[4:8,  0:8]
        if config['image_on']:
            self._gs_traj_visualizer    = self._gs[8:16, 0:4]
            self._gs_image_visualizer   = self._gs[8:16, 4:8]
        else:
            self._gs_traj_visualizer    = self._gs[8:16, 0:8]

        # Create GUI components.
        self._action_panel = ActionPanel(self._fig, self._gs_action_panel, 1, 4, actions_arr)
        self._action_output = Textbox(self._fig, self._gs_action_output, border_on=True)
        self._status_output = Textbox(self._fig, self._gs_status_output, border_on=False)
        self._algthm_output = Textbox(self._fig, self._gs_algthm_output,
                max_display_size=config['algthm_output_max_display_size'],
                log_filename=self._log_filename,
                fontsize=config['algthm_output_fontsize'],
                font_family='monospace')
        self._cost_plotter = MeanPlotter(self._fig, self._gs_cost_plotter,
                color='blue', label='mean cost')
        self._traj_visualizer = Plotter3D(self._fig, self._gs_traj_visualizer,
                num_plots=self._hyperparams['conditions'])
        if config['image_on']:
            self._image_visualizer = ImageVisualizer(self._fig,
                    self._gs_image_visualizer, cropsize=config['image_size'],
                    rostopic=config['image_topic'], show_overlay_buttons=True)

        # Setup GUI components.
        self._algthm_output.log_text('\n')
        self.set_output_text(self._hyperparams['info'])
        if config['initial_mode'] == 'run':
            self.run_mode()
        else:
            self.wait_mode()

        # Setup 3D Trajectory Visualizer plot titles and legends
        for m in range(self._hyperparams['conditions']):
            self._traj_visualizer.set_title(m, 'Condition %d' % (m))
        self._traj_visualizer.add_legend(linestyle='-', marker='None',
                color='green', label='Trajectory Samples')
        self._traj_visualizer.add_legend(linestyle='-', marker='None',
                color='blue', label='Policy Samples')
        self._traj_visualizer.add_legend(linestyle='None', marker='x',
                color=(0.5, 0, 0), label='LG Controller Means')
        self._traj_visualizer.add_legend(linestyle='-', marker='None',
                color='red', label='LG Controller Distributions')

        self._fig.canvas.draw()

        # Display calculating thread
        def display_calculating(delay, run_event):
            while True:
                if not run_event.is_set():
                    run_event.wait()
                if run_event.is_set():
                    self.set_status_text('Calculating.')
                    time.sleep(delay)
                if run_event.is_set():
                    self.set_status_text('Calculating..')
                    time.sleep(delay)
                if run_event.is_set():
                    self.set_status_text('Calculating...')
                    time.sleep(delay)

        self._calculating_run = threading.Event()
        self._calculating_thread = threading.Thread(target=display_calculating,
                args=(1, self._calculating_run))
        self._calculating_thread.daemon = True
        self._calculating_thread.start()

    # GPS Training functions
    def request_stop(self, event=None):
        self.request_mode('stop')

    def request_reset(self, event=None):
        self.request_mode('reset')

    def request_go(self, event=None):
        self.request_mode('go')

    def request_fail(self, event=None):
        self.request_mode('fail')

    def request_mode(self, request):
        """
        Sets the request mode (stop, reset, go, fail). The request is read by
        gps_main before sampling, and the appropriate action is taken.
        """
        self.mode = 'request'
        self.request = request
        self.set_action_text(self.request + ' requested')
        self.set_action_bgcolor(self._colors[self.request], alpha=0.2)

    def process_mode(self):
        """
        Completes the current request, after it is first read by gps_main.
        Displays visual confirmation that the request was processed,
        displays any error messages, and then switches into mode 'run' or 'wait'.
        """
        self.mode = 'process'
        self.set_action_text(self.request + ' processed')
        self.set_action_bgcolor(self._colors[self.request], alpha=1.0)
        if self.err_msg:
            self.set_action_text(self.request + ' processed' + '\nERROR: ' +
                                 self.err_msg)
            self.err_msg = None
            time.sleep(1.0)
        else:
            time.sleep(0.5)
        if self.request in ('stop', 'reset', 'fail'):
            self.wait_mode()
        elif self.request == 'go':
            self.run_mode()
        self.request = None

    def wait_mode(self):
        self.mode = 'wait'
        self.set_action_text('waiting')
        self.set_action_bgcolor(self._colors[self.mode], alpha=1.0)

    def run_mode(self):
        self.mode = 'run'
        self.set_action_text('running')
        self.set_action_bgcolor(self._colors[self.mode], alpha=1.0)

    def end_mode(self):
        self.mode = 'end'
        self.set_action_text('ended')
        self.set_action_bgcolor(self._colors[self.mode], alpha=1.0)

    def estop(self, event=None):
        self.set_action_text('estop: NOT IMPLEMENTED')

    # GUI functions
    def set_action_text(self, text):
        self._action_output.set_text(text)
        self._cost_plotter.draw_ticklabels()    # redraw overflow ticklabels

    def set_action_bgcolor(self, color, alpha=1.0):
        self._action_output.set_bgcolor(color, alpha)
        self._cost_plotter.draw_ticklabels()    # redraw overflow ticklabels

    def set_status_text(self, text):
        self._status_output.set_text(text)
        self._cost_plotter.draw_ticklabels()    # redraw overflow ticklabels

    def set_output_text(self, text):
        self._algthm_output.set_text(text)
        self._cost_plotter.draw_ticklabels()    # redraw overflow ticklabels

    def append_output_text(self, text):
        self._algthm_output.append_text(text)
        self._cost_plotter.draw_ticklabels()    # redraw overflow ticklabels

    def start_display_calculating(self):
        self._calculating_run.set()

    def stop_display_calculating(self):
        self._calculating_run.clear()

    def set_image_overlays(self, condition):
        """
        Sets up the image visualizer with what images to overlay if
        "overlay_initial_image" or "overlay_target_image" is pressed.
        """
        if not config['image_on'] or not self._target_filename:
            return
        initial_image = load_data_from_npz(self._target_filename,
                config['image_overlay_actuator'], str(condition),
                'initial', 'image', default=None)
        target_image  = load_data_from_npz(self._target_filename,
            config['image_overlay_actuator'], str(condition),
                'target',  'image', default=None)
        self._image_visualizer.set_initial_image(initial_image,
                alpha=config['image_overlay_alpha'])
        self._image_visualizer.set_target_image(target_image,
                alpha=config['image_overlay_alpha'])

    # Iteration update functions
    def update(self, itr, algorithm, agent, traj_sample_lists, pol_sample_lists):
        """
        After each iteration, update the iteration data output, the cost plot,
        and the 3D trajectory visualizations (if end effector points exist).
        """
        if self._first_update:
            self._output_column_titles(algorithm)
            self._first_update = False

        costs = [np.mean(np.sum(algorithm.prev[m].cs, axis=1)) for m in range(algorithm.M)]
        self._update_iteration_data(itr, algorithm, costs, pol_sample_lists)
        self._cost_plotter.update(costs, t=itr)
        if END_EFFECTOR_POINTS in agent.x_data_types:
            self._update_trajectory_visualizations(algorithm, agent,
                    traj_sample_lists, pol_sample_lists)

        self._fig.canvas.draw()
        self._fig.canvas.flush_events() # Fixes bug in Qt4Agg backend

    def _output_column_titles(self, algorithm, policy_titles=False):
        """
        Setup iteration data column titles: iteration, average cost, and for
        each condition the mean cost over samples, step size, linear Guassian
        controller entropies, and initial/final KL divergences for BADMM.
        """
        self.set_output_text(self._hyperparams['experiment_name'])
        # if isinstance(algorithm, AlgorithmMDGPS) or isinstance(algorithm, AlgorithmBADMM):
        #     condition_titles = '%3s | %8s %12s' % ('', '', '')
        #     itr_data_fields  = '%3s | %8s %12s' % ('itr', 'avg_cost', 'avg_pol_cost')
        # else:
        condition_titles = '%3s | %8s' % ('', '')
        itr_data_fields  = '%3s | %8s' % ('itr', 'avg_cost')
        for m in range(algorithm.M):
            condition_titles += ' | %8s %9s %-7d' % ('', 'condition', m)
            itr_data_fields  += ' | %8s %8s %8s' % ('  cost  ', '  step  ', 'entropy ')
            # if isinstance(algorithm, AlgorithmBADMM):
            #     condition_titles += ' %8s %8s %8s' % ('', '', '')
            #     itr_data_fields  += ' %8s %8s %8s' % ('pol_cost', 'kl_div_i', 'kl_div_f')
            # elif isinstance(algorithm, AlgorithmMDGPS):
            #     condition_titles += ' %8s' % ('')
            #     itr_data_fields  += ' %8s' % ('pol_cost')
        self.append_output_text(condition_titles)
        self.append_output_text(itr_data_fields)

    def _update_iteration_data(self, itr, algorithm, costs, pol_sample_lists):
        """
        Update iteration data information: iteration, average cost, and for
        each condition the mean cost over samples, step size, linear Guassian
        controller entropies, and initial/final KL divergences for BADMM.
        """
        avg_cost = np.mean(costs)
        if pol_sample_lists is not None:
            test_idx = algorithm._hyperparams['test_conditions']
            # pol_sample_lists is a list of singletons
            samples = [sl[0] for sl in pol_sample_lists]
            pol_costs = [np.sum(algorithm.cost[idx].eval(s)[0])
                    for s, idx in zip(samples, test_idx)]
            itr_data = '%3d | %8.2f %12.2f' % (itr, avg_cost, np.mean(pol_costs))
        else:
            itr_data = '%3d | %8.2f' % (itr, avg_cost)
        for m in range(algorithm.M):
            cost = costs[m]
            step = np.mean(algorithm.prev[m].step_mult * algorithm.base_kl_step)
            entropy = 2*np.sum(np.log(np.diagonal(algorithm.prev[m].traj_distr.chol_pol_covar,
                    axis1=1, axis2=2)))
            itr_data += ' | %8.2f %8.2f %8.2f' % (cost, step, entropy)
            # if isinstance(algorithm, AlgorithmBADMM):
            #     kl_div_i = algorithm.cur[m].pol_info.init_kl.mean()
            #     kl_div_f = algorithm.cur[m].pol_info.prev_kl.mean()
            #     itr_data += ' %8.2f %8.2f %8.2f' % (pol_costs[m], kl_div_i, kl_div_f)
            # elif isinstance(algorithm, AlgorithmMDGPS):
            #     # TODO: Change for test/train better.
            #     if test_idx == algorithm._hyperparams['train_conditions']:
            #         itr_data += ' %8.2f' % (pol_costs[m])
            #     else:
            #         itr_data += ' %8s' % ("N/A")
        self.append_output_text(itr_data)

    def _update_trajectory_visualizations(self, algorithm, agent,
                traj_sample_lists, pol_sample_lists):
        """
        Update 3D trajectory visualizations information: the trajectory samples,
        policy samples, and linear Gaussian controller means and covariances.
        """
        xlim, ylim, zlim = self._calculate_3d_axis_limits(traj_sample_lists, pol_sample_lists)
        for m in range(algorithm.M):
            self._traj_visualizer.clear(m)
            self._traj_visualizer.set_lim(i=m, xlim=xlim, ylim=ylim, zlim=zlim)
            if algorithm._hyperparams['fit_dynamics']:
                self._update_linear_gaussian_controller_plots(algorithm, agent, m)
            self._update_samples_plots(traj_sample_lists, m, 'green', 'Trajectory Samples')
            if pol_sample_lists:
                self._update_samples_plots(pol_sample_lists,  m, 'blue',  'Policy Samples')
        self._traj_visualizer.draw()    # this must be called explicitly

    def _calculate_3d_axis_limits(self, traj_sample_lists, pol_sample_lists):
        """
        Calculate the 3D axis limits shared between trajectory plots,
        based on the minimum and maximum xyz values across all samples.
        """
        all_eept = np.empty((0, 3))
        sample_lists = traj_sample_lists
        if pol_sample_lists:
            sample_lists += traj_sample_lists
        for sample_list in sample_lists:
            for sample in sample_list.get_samples():
                ee_pt = sample.get(END_EFFECTOR_POINTS)
                for i in range(ee_pt.shape[1]//3):
                    ee_pt_i = ee_pt[:, 3*i+0:3*i+3]
                    all_eept = np.r_[all_eept, ee_pt_i]
        min_xyz = np.amin(all_eept, axis=0)
        max_xyz = np.amax(all_eept, axis=0)
        xlim = buffered_axis_limits(min_xyz[0], max_xyz[0], buffer_factor=1.25)
        ylim = buffered_axis_limits(min_xyz[1], max_xyz[1], buffer_factor=1.25)
        zlim = buffered_axis_limits(min_xyz[2], max_xyz[2], buffer_factor=1.25)
        return xlim, ylim, zlim

    def _update_linear_gaussian_controller_plots(self, algorithm, agent, m):
        """
        Update the linear Guassian controller plots with iteration data,
        for the mean and covariances of the end effector points.
        """
        # Calculate mean and covariance for end effector points
        eept_idx = agent.get_idx_x(END_EFFECTOR_POINTS)
        start, end = eept_idx[0], eept_idx[-1]
        mu, sigma = algorithm.traj_opt.forward(algorithm.prev[m].traj_distr, algorithm.prev[m].traj_info)
        mu_eept, sigma_eept = mu[:, start:end+1], sigma[:, start:end+1, start:end+1]

        # Linear Gaussian Controller Distributions (Red)
        #type(mu_eept.shape[1])
        print(mu_eept.shape[1])
        print(mu_eept.shape[1]//3)
        for i in range(mu_eept.shape[1]//3):
            mu, sigma = mu_eept[:, 3*i+0:3*i+3], sigma_eept[:, 3*i+0:3*i+3, 3*i+0:3*i+3]
            self._traj_visualizer.plot_3d_gaussian(i=m, mu=mu, sigma=sigma,
                    edges=100, linestyle='-', linewidth=1.0, color='red',
                    alpha=0.15, label='LG Controller Distributions')

        # Linear Gaussian Controller Means (Dark Red)
        for i in range(mu_eept.shape[1]//3):
            mu = mu_eept[:, 3*i+0:3*i+3]
            self._traj_visualizer.plot_3d_points(i=m, points=mu, linestyle='None',
                    marker='x', markersize=5.0, markeredgewidth=1.0,
                    color=(0.5, 0, 0), alpha=1.0, label='LG Controller Means')

    def _update_samples_plots(self, sample_lists, m, color, label):
        """
        Update the samples plots with iteration data, for the trajectory samples
        and the policy samples.
        """
        samples = sample_lists[m].get_samples()
        for sample in samples:
            ee_pt = sample.get(END_EFFECTOR_POINTS)
            for i in range(ee_pt.shape[1]//3):
                ee_pt_i = ee_pt[:, 3*i+0:3*i+3]
                self._traj_visualizer.plot_3d_points(m, ee_pt_i, color=color, label=label)

    def save_figure(self, filename):
        self._fig.savefig(filename)
