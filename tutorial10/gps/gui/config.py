""" Default configuration and hyperparameter values for GUI objects. """
import itertools

#from gps.proto.gps_pb2 import TRIAL_ARM, AUXILIARY_ARM
from gps_pb2 import TRIAL_ARM, AUXILIARY_ARM
from gps.gui.ps3_config import PS3_BUTTON, INVERTED_PS3_BUTTON


# Mappings from actions to their corresponding keyboard bindings.
# WARNING: keybindings must be unique
keyboard_bindings = {
    # Target Setup.
    'ptn': 'left',  # previous target number
    'ntn': 'right', # next target number
    'pat': 'down',  # previous actuator type
    'nat': 'up',    # next actuator type

    'sip': 'i',     # set initial position
    'stp': 't',     # set target position
    'sii': 'z',     # set initial image
    'sti': 'x',     # set target image

    'mti': 'm',     # move to initial
    'mtt': 'n',     # move to target
    'rc': 'c',      # relax controller
    'mm': 'q',      # mannequin mode

    # GPS Training.
    'stop' : 's',   # stop
    'reset': 'r',   # reset
    'go'   : 'g',   # go
    'fail' : 'f',   # fail

    # Image Visualizer
    'oii'  : 'o',   # overlay initial image
    'oti'  : 'p',   # overlay target image
}
inverted_keyboard_bindings = {value: key
                              for key, value in keyboard_bindings.items()}
#                              for key, value in keyboard_bindings.iteritems()}

# Mappings from actions to their corresponding PS3 controller bindings.
ps3_bindings = {
    # Target Setup
    'ptn': (PS3_BUTTON['rear_right_1'], PS3_BUTTON['cross_left']),
    'ntn': (PS3_BUTTON['rear_right_1'], PS3_BUTTON['cross_right']),
    'pat': (PS3_BUTTON['rear_right_1'], PS3_BUTTON['cross_down']),
    'nat': (PS3_BUTTON['rear_right_1'], PS3_BUTTON['cross_up']),

    'sip': (PS3_BUTTON['rear_right_1'], PS3_BUTTON['action_square']),
    'stp': (PS3_BUTTON['rear_right_1'], PS3_BUTTON['action_circle']),
    'sii': (PS3_BUTTON['rear_right_1'], PS3_BUTTON['action_cross']),
    'sti': (PS3_BUTTON['rear_right_1'], PS3_BUTTON['action_triangle']),

    'mti': (PS3_BUTTON['rear_right_2'], PS3_BUTTON['cross_left']),
    'mtt': (PS3_BUTTON['rear_right_2'], PS3_BUTTON['cross_right']),
    'rc' : (PS3_BUTTON['rear_right_2'], PS3_BUTTON['cross_down']),
    'mm' : (PS3_BUTTON['rear_right_2'], PS3_BUTTON['cross_up']),

    # GPS Training
    'stop' : (PS3_BUTTON['rear_right_2'], PS3_BUTTON['action_square']),
    'reset': (PS3_BUTTON['rear_right_2'], PS3_BUTTON['action_triangle']),
    'go'   : (PS3_BUTTON['rear_right_2'], PS3_BUTTON['action_circle']),
    'fail' : (PS3_BUTTON['rear_right_2'], PS3_BUTTON['action_cross']),

    # Image Visualizer
    'oii'  : (PS3_BUTTON['cross_up']    ,),
    'oti'  : (PS3_BUTTON['cross_down']  ,),
}
#inverted_ps3_bindings = {value: key for key, value in ps3_bindings.iteritems()}
inverted_ps3_bindings = {value: key for key, value in ps3_bindings.items()}

permuted_inverted_ps3_bindings = {}
#for key, value in list(inverted_ps3_bindings.iteritems()):
for key, value in list(inverted_ps3_bindings.items()):
    for permuted_key in itertools.permutations(key, len(key)):
        permuted_inverted_ps3_bindings[permuted_key] = value

config = {
    # Keyboard shortcuts bindings
    'keyboard_bindings': keyboard_bindings,
    'inverted_keyboard_bindings': inverted_keyboard_bindings,

    # PS3 controller bindings
    'ps3_topic': 'joy',
    'ps3_process_rate': 20,  # Only process 1/20 of PS3 messages.
    'ps3_button': PS3_BUTTON,
    'inverted_ps3_button': INVERTED_PS3_BUTTON,
    'ps3_bindings': ps3_bindings,
    'inverted_ps3_bindings': inverted_ps3_bindings,
    'permuted_inverted_ps3_bindings': permuted_inverted_ps3_bindings,

    # Images
    'image_on': True,
    'image_topic': '/camera/rgb/image_color',
    'image_size': (240, 240),
    'image_overlay_actuator': 'trial_arm',
    'image_overlay_alpha': 0.3,

    # Both GUIs
    'figsize': (12, 12),

    # Target Setup
    'num_targets': 10,
    'actuator_types': [TRIAL_ARM, AUXILIARY_ARM],
    'actuator_names': ['trial_arm', 'auxiliary_arm'],
    'target_output_fontsize': 10,

    # GPS Training
    'initial_mode': 'run',
    'algthm_output_fontsize': 10,
    'algthm_output_max_display_size': 15,
}

def generate_experiment_info(config):
    """
    Generate experiment info, to be displayed by GPS Trainig GUI.
    Assumes config is the config created in hyperparams.py
    """
    common = config['common']
    algorithm = config['algorithm']

    if type(algorithm['cost']) == list:
        algorithm_cost_type = algorithm['cost'][0]['type'].__name__
        if (algorithm_cost_type) == 'CostSum':
            algorithm_cost_type += '(%s)' % ', '.join(
                    map(lambda cost: cost['type'].__name__,
                        algorithm['cost'][0]['costs']))
    else:
        algorithm_cost_type = algorithm['cost']['type'].__name__
        if (algorithm_cost_type) == 'CostSum':
            algorithm_cost_type += '(%s)' % ', '.join(
                    map(lambda cost: cost['type'].__name__,
                        algorithm['cost']['costs']))

    if 'dynamics' in algorithm:        
        alg_dyn = str(algorithm['dynamics']['type'].__name__)
    else:
        alg_dyn = 'None'       

    return (
        'exp_name:   ' + str(common['experiment_name'])              + '\n' +
        'alg_type:   ' + str(algorithm['type'].__name__)             + '\n' +
        'alg_dyn:    ' + alg_dyn + '\n' +
        'alg_cost:   ' + str(algorithm_cost_type)                    + '\n' +
        'iterations: ' + str(config['iterations'])                   + '\n' +
        'conditions: ' + str(algorithm['conditions'])                + '\n' +
        'samples:    ' + str(config['num_samples'])                  + '\n'
    )
