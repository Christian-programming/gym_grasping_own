"""
Sample Random Poses.
"""
import math
import numpy as np


class RandomPoseSamplerReal:
    """ Sample random poses """
    def __init__(self,
                 num_samples=10,
                 center=np.array([0, -0.55, 0.2]),
                 theta_limits=(0, 2 * math.pi),
                 r_limits=(0.05, 0.15),
                 h_limits=(0, 0.1),
                 trans_limits=(-0.1, 0.1),
                 rot_limits=(-np.radians(10), np.radians(10)),
                 pitch_limit=(-np.radians(10), np.radians(10)),
                 roll_limit=(-np.radians(10), np.radians(10))):
        self.center = center
        self.theta_limits = theta_limits
        self.r_limits = r_limits
        self.h_limits = h_limits
        self.trans_limits = trans_limits
        self.rot_limits = rot_limits
        self.pitch_limit = pitch_limit
        self.roll_limit = roll_limit
        self.num_samples = num_samples

    def sample_pose(self):
        '''sample a  random pose'''
        theta = np.random.uniform(*self.theta_limits)
        rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]])
        vec = rot_mat @ np.array([1, 0, 0])
        rand_unif = np.random.uniform(*self.r_limits)
        vec = vec * rand_unif
        theta_offset = np.random.uniform(*self.rot_limits)
        trans = np.cross(np.array([0, 0, 1]), vec)
        trans = trans * np.random.uniform(*self.trans_limits)
        height = np.array([0, 0, 1]) * np.random.uniform(*self.h_limits)
        trans_final = self.center + vec + trans + height
        pitch = np.random.uniform(*self.pitch_limit)
        roll = np.random.uniform(*self.roll_limit)
        return (*trans_final, math.pi + pitch, roll, theta + math.pi / 2 + theta_offset)


class RandomPoseSampler(RandomPoseSamplerReal):
    """ Sample random poses for simulation """
    def __init__(self,
                 num_samples=10,
                 center=np.array([0, -0.55, 0.4]),
                 theta_limits=(0, 2 * math.pi),
                 r_limits=(0.05, 0.15),
                 h_limits=(0, 0.1),
                 trans_limits=(-0.1, 0.1),
                 rot_limits=(-np.radians(10), np.radians(10)),
                 pitch_limit=(-np.radians(10), np.radians(10)),
                 roll_limit=(-np.radians(10), np.radians(10))):
        self.center = center
        self.theta_limits = theta_limits
        self.r_limits = r_limits
        self.h_limits = h_limits
        self.trans_limits = trans_limits
        self.rot_limits = rot_limits
        self.pitch_limit = pitch_limit
        self.roll_limit = roll_limit
        self.num_samples = num_samples
