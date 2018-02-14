#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import numpy as np
import sympy as sp
import time

import dh 
import jacobian
import inverse_kinematics
import utils

pwm = utils.import_pwm()

class Planner(object):
    '''
    This class selects actions for the manipulator
    This is extremely limited right now, for example 
        - this doesn't account for constraints
        - it only outputs a single angle to achieve
    '''

    def __init__(self, params):
        '''
        Args:
            - params: DH params defining robot links
                dictionary of dictionaries
                first set of keys is 1 up to 6 indicating the link number 
                each link dictionary contains a map from string to sympy symbol
                valid strings are a,l,t,d 
                a = alpha, l = length, t = theta, d = distance 
        '''
        self.params = params
        # forward kinematics
        self.T, self.transforms = dh.build_transforms(params)
        # extract joint configuration
        self.qs = utils.extract_joints_from_params(params)
        self.nq = len(self.qs)
        # compute jacobian
        self.J = jacobian.jacobian(self.transforms, self.qs)
        # compute inverse of jacobian (this takes a while)
        self.J_inv = sp.zeros(6,6)
        self.J_inv[:self.nq,:] = utils.left_pseudoinv(self.J[:,:self.nq])

    def find_joint_config(self, p_des, q_cur):
        '''
        Description:
            - given an end-effector position and orientation vector 'pos', first 
            finds the joint configuration yielding that position, then sets 
            servos to achieve that configuration.

        Args:
            - p_des: 6-vector of floats. First 3 indicate x,y,z position, second 3
            indicate α, β, γ orientation. The desired state.
            - q_cur: current joint configuration. 6-vector of floats.
        '''
        # increase desired position to include any missing coordinates as zeros
        if len(p_des) < 6:
            zeros = np.zeros(6 - len(p_des))
            p_des = np.concatenate((p_des, zeros))

        # compute the joint angles that achieve the desired position
        p_des = sp.Matrix(p_des)
        q_cur = sp.Matrix(q_cur)
        q_des, valid = inverse_kinematics.find_joint_config(
            self.J,
            self.J_inv,
            self.qs,
            p_des,
            q_cur,
            self.T
        )
        # assume joints are all revolute and that their angles should be 
        # in the range [0,2pi], so clip them here
        q_des = utils.normalize_angles_0to2pi(q_des)
        return q_des

    def forward_kinematics(self, q):
        p = self.T.evalf(subs=dict(zip(self.qs, q)))[:-1,-1]
        return np.array(p).astype(float).reshape(-1)

class Servo(object):
    '''
    A servo on the manipulator
    '''
    def __init__(
            self, 
            channel, 
            min_angle=10,
            max_angle=170,
            min_pulse=150,
            max_pulse=600):
        self.channel = channel
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.min_pulse = min_pulse
        self.max_pulse = max_pulse

    def set_state(self, pwm, rad):
        utils.set_servo_angle(
            pwm,
            self.channel,
            np.rad2deg(rad),
            self.min_angle,
            self.max_angle,
            self.min_pulse,
            self.max_pulse
        )

    def random_state(self):
        return np.deg2rad(
            self.min_angle + np.random.rand() * (self.max_angle - self.min_angle)
        )

class Controller(object):
    '''
    This class enacts the actions of the planner through communication with 
    the servos on the manipulator
    '''

    def __init__(self, servos, address=0x40, debug=False):
        # Initialise the PWM device
        self.pwm = pwm.PWM(address)
        self.servos = servos
        self.debug = debug
        # initial end-effector position
        self.state = sp.zeros(6,1)

    def set_end_effector_pos(self, new_state):
        for i, servo in enumerate(self.servos):
            servo.set_state(self.pwm, new_state[i])
            self.state[i] = new_state[i]
        if self.debug:
            print('controller state: {}'.format(self.state))

    def random_state(self):
        qs = []
        for servo in self.servos:
            q = servo.random_state()
            qs.append(q)
        return qs

class Manipulator(object):
    '''
    Represents the manipulator and coordinates the planner and controller
    '''

    def __init__(self, planner, controller):
        self.planner = planner
        self.controller = controller

    def set_end_effector_position(self, pos):
        new_state = self.planner.find_joint_config(pos, self.controller.state)
        self.controller.set_end_effector_pos(new_state)

    def random_position(self):
        '''
        Returns a random end-effector position
        '''
        # sample valid joint configuration
        q = self.controller.random_state()
        # determine the position associated with this joint configuration
        pos = self.planner.forward_kinematics(q)
        return pos

def build_RR_manipulator():
    # symbols used throughout equations
    t1, t2 = sp.symbols('t1 t2')

    # manipulator constants
    l1 = 1
    l2 = 1

    # compute transformation matrix
    params = dict()
    params[1] = collections.defaultdict(int, dict(t=t1))
    params[2] = collections.defaultdict(int, dict(l=l1, t=t2))
    params[3] = collections.defaultdict(int, dict(l=l2))

    # build manipulator
    planner = Planner(params)

    # build controller
    servos = [Servo(i) for i in range(2)]
    controller = Controller(servos)

    # build manipulator
    manipulator = Manipulator(planner, controller)
    return manipulator