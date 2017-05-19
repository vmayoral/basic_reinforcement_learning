from __future__ import division

import os.path
from datetime import datetime
import numpy as np

#from gps import __file__ as gps_filepath
#from gps.agent.box2d.agent_box2d import AgentBox2D
#from gps.agent.box2d.arm_world import ArmWorld
#from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
#from gps.algorithm.cost.cost_state import CostState
#from gps.algorithm.cost.cost_action import CostAction
#from gps.algorithm.cost.cost_sum import CostSum
#from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
#from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
#from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
#from gps.algorithm.policy.lin_gauss_init import init_lqr
#from gps.gui.config import generate_experiment_info
#from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, ACTION
from gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, ACTION

SENSOR_DIMS = {
    JOINT_ANGLES: 2,
    JOINT_VELOCITIES: 2,
    END_EFFECTOR_POINTS: 3,
    ACTION: 2
}

from gps import __file__ as gps_filepath
gps_filepath = os.path.abspath(gps_filepath)
print(gps_filepath)


BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/box2d_arm_example/'


common = {
    'experiment_name': 'box2d_arm_example' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 1,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentBox2D,
    'target_state' : np.array([0, 0]),
    'world' : ArmWorld,
    'render' : False,
    'x0': np.array([0.75*np.pi, 0.5*np.pi, 0, 0, 0, 0, 0]),
    'rk': 0,
    'dt': 0.05,
    'substeps': 1,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([]),
    'pos_body_offset': np.array([]),
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS],
    'obs_include': [],
}

algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains': np.zeros(SENSOR_DIMS[ACTION]),
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 0.1,
    'stiffness': 0.01,
    'dt': agent['dt'],
    'T': agent['T'],
}

action_cost = {
    'type': CostAction,
    'wu': np.array([1, 1])
}

state_cost = {
    'type': CostState,
    'data_types' : {
        JOINT_ANGLES: {
            'wp': np.array([1, 1]),
            'target_state': agent["target_state"],
        },
    },
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [action_cost, state_cost],
    'weights': [1e-5, 1.0],
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {}

config = {
    'iterations': 10,
    'num_samples': 5,
    'verbose_trials': 5,
    'common': common,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
}

common['info'] = generate_experiment_info(config)
