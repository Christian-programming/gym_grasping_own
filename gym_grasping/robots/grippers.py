"""
Code for simulated grippers.
"""
import math
import numpy as np
import pybullet as p
gS = 1

# This is different for the suction gripper at the moment, should be merged
GRIPPER_OPEN = 1
GRIPPER_CLOSE = -1
GRIPPER_DEFAULT = 1


class WSG50:
    """
    Weiss Robotics WSG50 gripper.

    act_type:
        'discrete' -> map continuous inputs to open/close
        'continuous' -> allow intermediary opening states
    """

    def __init__(self, cid, robot, env_params, calibration, gripper_act_type='discrete', **kwargs):
        # indexes relative to 7 joint kuka
        # TODO(max) input scaling relative to gripper2 env, make [0,1]
        # normalized

        # todo: pass this through explicitly
        self.p = p
        self.cid = cid
        self.robot = robot
        self.joint_ids = [8, 9]
        self.finger_link_ids = [8, 9]
        finger_tip_force = 20 * gS
        self.hand_forces = (finger_tip_force, finger_tip_force)
        self.max_dist = 0.005 * gS
        self.grasp_action_u = lambda s: [0, 0, 0.001, 0, 0.005]
        self.grasp_action_a = lambda s, pct: [0, 0, 0.000, 0, 0.010 * (.5-pct)]
        self.grasp_action_b = lambda s, pct: [0, 0, 0.005, 0, -.005]
        self.act_type = gripper_act_type
        self.closing_threshold = 0
        self.prev_action = 1
        # Sampled variables, set in reset
        self.action_delay = None
        self.action_queue = None
        self.speed = None
        self.rot_vel = None
        #
        self.open_action = GRIPPER_OPEN
        self.close_action = GRIPPER_CLOSE
        #
        # randomized parameters
        self.params = env_params
        gripper_join_ul = calibration["gripper_joint_ul"]
        self.params.add_variable('gripper_joint_ul', center=gripper_join_ul, d=0.001)
        # reset pose
        self.params.add_variable("gripper_rot", center=0, d=math.pi / 12)
        self.params.add_variable("gripper_rot_vel", 6)
        self.params.add_variable("gripper_speed", center=10, d=0.1)
        # this number can be large 3(video delay) * 4(frameskip) = 12
        self.params.add_variable("gripper_delay", center=12)

        # TODO(max): How is the flange define, maybe this should be positive?
        self.T_tcp_flange = np.array(
           [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, -0.26],
            [0, 0, 0, 1]])

    def reset(self, gripper_open=True):
        """
        Reset gripper, resets action queue and sampled variables.
        """
        self.speed = self.params.gripper_speed
        self.action_delay = int(self.params.gripper_delay)

        initial_vals = 1 if gripper_open else -1
        self.action_queue = [initial_vals] * self.action_delay
        self.rot_vel = self.params.gripper_rot_vel

    def reset_pose(self, norm_action=GRIPPER_DEFAULT):
        hand_poses, _ = self.get_control(norm_action)
        for i, pos in enumerate(hand_poses):
            self.p.resetJointState(self.robot.robot_uid, self.joint_ids[i],
                                   pos, physicsClientId=self.cid)

    def get_opening_width(self):
        """
        opening width for state variable.
        """
        robot_uid = self.robot.robot_uid
        width = p.getJointState(robot_uid, self.joint_ids[0], physicsClientId=self.cid)[0] \
            + p.getJointState(robot_uid, self.joint_ids[1], physicsClientId=self.cid)[0]
        # gripper doesn't close completely, so subtract lower joint limit
        width -= 2 * p.getJointInfo(robot_uid, self.joint_ids[0], physicsClientId=self.cid)[8]
        return width

    def get_control(self, norm_action):
        """
        imput:
            normed action
        ouput:
            joint poses, joint forces
        """
        self.action_queue.append(norm_action)
        return self._get_control(self.action_queue.pop(0))

    def _get_control(self, norm_action):
        # environment action space is discrete or multi-discrete
        if isinstance(norm_action, np.int8) and norm_action == 0:
            norm_action = 1
        # environment action space is continuous
        else:
            if self.act_type == 'discrete':
                if norm_action >= self.closing_threshold:
                    norm_action = 1
                else:
                    norm_action = -1
            else:
                # take a continouos variable as input, close if over threshold
                # this mirrors current hardware controler with binary
                # open/close
                norm_action = np.clip(norm_action, self.closing_threshold, 1)
                norm_action -= self.closing_threshold
                norm_action /= (1 - self.closing_threshold)
                norm_action *= 2
                norm_action -= 1

        a = (norm_action + 1) / 2 * self.params.gripper_joint_ul
        # fingerAngle = .4*norm_action + -.2 # [-1, 1] -> [.1,.3]
        hand_poses = (a, a)
        self.prev_action = norm_action
        return hand_poses, self.hand_forces


class SuctionGripper:
    """
    Suction gripper.
    """

    def __init__(self, env_params, act_type='continous'):
        assert act_type == 'continous'
        self.joint_ids = []
        self.joint_ids_active = []
        self.finger_link_ids = [8]
        self.open_action = 0
        self.close_action = 1

        self.T_tcp_flange = np.array(
           [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, -.08],
            [0, 0, 0, 1]])

        # TOOD(max): remove the need for this
        self.rot_vel = 6

        env_params.add_variable("gripper_rot", center=0, d=math.pi / 12)

    def reset(self):
        '''rese the gripper class state'''

    def get_control(self, norm_action):
        """
        imput:
            normed action
        ouput:
            joint poses, joint forces
        """
        return [[], 0]

    def reset_pose(self, norm_action=GRIPPER_DEFAULT):
        # nothing to do for suction gripper
        pass

    def get_state(self, gripper_action=None):
        """
        imput:
            gripper action
        ouput:
            open or close
        """
        if gripper_action <= self.open_action:
            return "open"
        return "close"

    # TODO(lukas): rename get_state or get_obs?
    def get_opening_width(self):
        '''for propriocptive state'''
        return [0]
