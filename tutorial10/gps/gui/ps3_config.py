### DO NOT CHANGE THE CONTENTS OF THIS FILE ###
# PS3 Joystick Buttons and Axes
# documentation: http://wiki.ros.org/ps3joy

# Mappings from PS3 buttons to their corresponding array indices.
PS3_BUTTON = {
    'select': 0,
    'stick_left': 1,
    'stick_right': 2,
    'start': 3,
    'cross_up': 4,
    'cross_right': 5,
    'cross_down': 6,
    'cross_left': 7,
    'rear_left_2': 8,
    'rear_right_2': 9,
    'rear_left_1': 10,
    'rear_right_1': 11,
    'action_triangle': 12,
    'action_circle': 13,
    'action_cross': 14,
    'action_square': 15,
    'pairing': 16,
}
#INVERTED_PS3_BUTTON = {value: key for key, value in PS3_BUTTON.iteritems()}
INVERTED_PS3_BUTTON = {value: key for key, value in PS3_BUTTON.items()}

# Mappings from PS3 axes to their corresponding array indices.
PS3_AXIS = {
    'stick_left_leftwards': 0,
    'stick_left_upwards': 1,
    'stick_right_leftwards': 2,
    'stick_right_upwards': 3,
    'button_cross_up': 4,
    'button_cross_right': 5,
    'button_cross_down': 6,
    'button_cross_left': 7,
    'button_rear_left_2': 8,
    'button_rear_right_2': 9,
    'button_rear_left_1': 10,
    'button_rear_right_1': 11,
    'button_action_triangle': 12,
    'button_action_circle': 13,
    'button_action_cross': 14,
    'button_action_square': 15,
    'acceleratometer_left': 16,
    'acceleratometer_forward': 17,
    'acceleratometer_up': 18,
    'gyro_yaw': 19,
}
#INVERTED_PS3_AXIS = {value: key for key, value in PS3_AXIS.iteritems()}
INVERTED_PS3_AXIS = {value: key for key, value in PS3_AXIS.items()}
