"""
PR2 Gripper class
"""
import numpy as np


class PR2Gripper:
    '''PR2 Gripper class'''
    def __init__(self, joint_ids, joint_ids_active, act_type='continuous'):
        self.nhp = 4
        self.joint_ids = joint_ids
        self.joint_ids_active = joint_ids_active
        self.finger_link_ids = [joint_ids[2], joint_ids[3]]
        self.finger_tip_force = 2
        self.hand_forces = [self.finger_tip_force] * 4
        self.act_type = act_type
        self.closing_threshold = 0.6
        self.upper_joint_limit = 0.5

    def get_control(self, finger_angle):
        """
        imput:
            normed action
        ouput:
            joint poses, joint forces
        """
        if isinstance(finger_angle, np.int8) and finger_angle == 0:
            finger_angle = 1
        else:
            if self.act_type == 'discrete':
                if finger_angle > self.closing_threshold:
                    finger_angle = 1
                else:
                    finger_angle = -1
            else:
                # open gripper after threshold (closing bias)
                finger_angle = np.clip(finger_angle, self.closing_threshold, 1)
                finger_angle -= self.closing_threshold
                finger_angle /= (1 - self.closing_threshold)
                finger_angle *= 2
                finger_angle -= 1
        m_fa = (finger_angle + 1) / 2 * self.upper_joint_limit
        # mFa = (finger_angle+1) * 0.25
        hand_poses = [m_fa, m_fa, m_fa, m_fa]
        return hand_poses, self.hand_forces
