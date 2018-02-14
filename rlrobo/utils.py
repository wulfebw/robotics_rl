
import importlib
import numpy as np
import sympy as sp
import sys

''' sympy utils '''
def extract_joints_from_params(params):
    links = sorted(params.keys())
    qs = []
    for link in links:
        for (attr, val) in params[link].items():
            if attr.startswith('d') or attr.startswith('t'):
                qs.append(val)
                break
    return qs

def left_pseudoinv(m):
    return sp.simplify((m.T * m) ** -1 * m.T)

def cumprod(ts):
    '''
    Description:
        - computes the cumulative product of a list of matrices

    Args:
        - ts: list of matrices, all square, all the same shape
    
    '''
    if len(ts) == 0:
        return []
    shape = ts[0].shape
    assert len(shape) == 2
    assert shape[0] == shape[1]
    side = shape[0]

    ret = []
    cur = sp.eye(side)
    for t in ts:
        cur = sp.simplify(cur * t)
        ret.append(cur)
    return ret

''' manipulator utils '''
def normalize_angles_0to2pi(angles):
    angles = [a % (2 * np.pi) for a in angles]
    return angles

def import_pwm():
    if sys.platform == 'linux2':
        # raspberry pi, import the adafruit pwm
        pwm_module = importlib.import_module('Adafruit_PWM_Servo_Driver')
    else:
        # import the mock pwm
        pwm_module = importlib.import_module('rlrobo.mock_pwm')
    return pwm_module

def set_servo_angle(
        pwm, 
        channel, 
        angle, 
        min_angle, 
        max_angle,
        min_pulse, 
        max_pulse):
    '''
    Description:
      - set servo angle
      - max_angle corresponds to maximum position clockwise with servo facing upward
    '''
    # clip angle to valid range
    angle = max(min(angle, max_angle), min_angle)

    # linearly interpolate between min_pulse and max_pulse
    # depending on where in the possible range angle falls
    frac = (angle - min_angle) / (max_angle - min_angle)
    pulse = int(min_pulse + frac * (max_pulse - min_pulse))
    pwm.setPWM(channel, 0, pulse)
