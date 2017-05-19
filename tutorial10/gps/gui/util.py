import os
import numpy as np

DEFAULT_JOINT_ANGLES = np.zeros(7)
DEFAULT_END_EFFECTOR_POSITIONS = np.zeros(3)
DEFAULT_END_EFFECTOR_ROTATIONS = np.zeros((3, 3))

def buffered_axis_limits(amin, amax, buffer_factor=1.0):
    """
    Increases the range (amin, amax) by buffer_factor on each side
    and then rounds to precision of 1/10th min or max.
    Used for generating good plotting limits.
    For example (0, 100) with buffer factor 1.1 is buffered to (-10, 110)
    and then rounded to the nearest 10.
    """
    diff = amax - amin
    amin -= (buffer_factor-1)*diff
    amax += (buffer_factor-1)*diff
    magnitude = np.floor(np.log10(np.amax(np.abs((amin, amax)) + 1e-100)))
    precision = np.power(10, magnitude-1)
    amin = np.floor(amin/precision) * precision
    amax = np.ceil (amax/precision) * precision
    return (amin, amax)

def save_pose_to_npz(filename, actuator_name, target_number, data_time, pose):
    """
    Saves a pose for the specified actuator name, target number, and data time.
    Args:
        filename - the target file ('../target.npz')
        actuator_name - the actuator name ('trial_arm', 'auxiliary_arm', etc.)
        target_number - the target number ('0', '1', '2', etc.)
        data_time - either 'initial or 'final'
        pose - (joint angle, end effector position, end effector rotation) tuple
    Return:
        None
    """
    ja, ee_pos, ee_rot = pose
    save_data_to_npz(filename, actuator_name, target_number, data_time,
                     'ja', ja)
    save_data_to_npz(filename, actuator_name, target_number, data_time,
                     'ee_pos', ee_pos)
    save_data_to_npz(filename, actuator_name, target_number, data_time,
                     'ee_rot', ee_rot)


def save_data_to_npz(filename, actuator_name, target_number, data_time,
                     data_name, value):
    """
    Save data to the specified file with key
    (actuator_name, target_number, data_time, data_name).
    """
    key = '_'.join((actuator_name, target_number, data_time, data_name))
    save_to_npz(filename, key, value)


def save_to_npz(filename, key, value):
    """
    Save a (key,value) pair to a npz dictionary.
    Args:
        filename: The file containing the npz dictionary.
        key: The key (string).
        value: The value (numpy array).
    """
    tmp = {}
    if os.path.exists(filename):
        with np.load(filename) as f:
            tmp = dict(f)
    tmp[key] = value
    np.savez(filename, **tmp)


def load_pose_from_npz(filename, actuator_name, target_number, data_time,
        default_ja=DEFAULT_JOINT_ANGLES,
        default_ee_pos=DEFAULT_END_EFFECTOR_POSITIONS,
        default_ee_rot=DEFAULT_END_EFFECTOR_ROTATIONS):
    """
    Loads a pose for the specified actuator name, target number, and data time.
    Args:
        filename - the target file ('../target.npz')
        actuator_name - the actuator name ('trial_arm', 'auxiliary_arm', etc.)
        target_number - the target number ('0', '1', '2', etc.)
        data_time - either 'initial or 'final'
    Return:
        pose - (joint angle, end effector position, end effector rotation) tuple 
    """
    ja = load_data_from_npz(filename, actuator_name, target_number, data_time,
                            'ja', default=default_ja)
    ee_pos = load_data_from_npz(filename, actuator_name, target_number,
                                data_time, 'ee_pos', default=default_ee_pos)
    ee_rot = load_data_from_npz(filename, actuator_name, target_number,
                                data_time, 'ee_rot', default=default_ee_rot)
    return (ja, ee_pos, ee_rot)


def load_data_from_npz(filename, actuator_name, target_number, data_time,
                       data_name, default=None):
    """
    Load data from the specified file with key
    (actuator_name, target_number, data_time, data_name).
    """
    key = '_'.join((actuator_name, target_number, data_time, data_name))
    return load_from_npz(filename, key, default)


def load_from_npz(filename, key, default=None):
    """
    Load a (key,value) pair from a npz dictionary. Returns default if failed.
    Args:
        filename: The file containing the npz dictionary.
        key: The key (string).
        value: The default value to return, if key or file not found.
    """
    try:
        with np.load(filename) as f:
            return f[key]
    except (IOError, KeyError):
        pass
    return default
