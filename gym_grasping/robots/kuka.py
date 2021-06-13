"""
Robot class for a simulated Kuka iiwa.
"""

import math
import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation as R

from gym_grasping.utils import state2matrix
from gym_grasping.robots.grippers import SuctionGripper, WSG50
from gym_grasping.robots.models import get_robot_path as rpath
from gym_grasping.robots.robots import Robot


class Kuka(Robot):
    """
    KUKA iiwa class

    See Robot class for args.
    """

    calibration = "kuka_iiwa/calibration_latest.json"

    def __init__(self, *args, calibration=None, **kwargs):
        # this sets env_params
        super().__init__(*args, **kwargs)
        #
        # gripped objects
        self.connected = set()
        #
        # initial pose
        self.initial_pose = calibration["rest_pose"]
        #
        # randomized parameters
        self.params.add_variable("robot_base", tag="geom",
                                 center=(-.1, 0, 0.07),
                                 d=(0, 0, 0.005))
        self.params.add_variable("robot_dv", tag="dyn", center=.001, d=.0002)
        self.params.add_variable("robot_drot", .01, tag="dyn")
        self.params.add_variable("joint_vel", 1, tag="dyn")
        self.params.add_variable("max_rot_diff", 0.09, tag="dyn")

    def load(self, workspace_offset=None):
        joint_names = ['J0', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6',
                       'gripper_to_arm', 'base_left_finger_joint',
                       'base_right_finger_joint']

        self.base_orientation = self.p.getQuaternionFromEuler([0, 0, 0])
        # robot workspace_offset, in world cooridinates
        if workspace_offset is None:
            workspace_offset = [[-0.25, -0.1, 0.295],
                                [0.25, 0.21, 0.465]]  # xyz high
        self.workspace = np.array(workspace_offset) + self.params.table_pos
        self.workspace_offset = workspace_offset
        # load_urdf, this populates joint_dict and active_joint_dict
        super().load(joint_names)

        arm_joint_names = ['J0', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6']
        self.all_joint_ids = [self.joint_dict[jn] for jn in arm_joint_names]
        active_jids = [self.active_joint_dict[jn] for jn in arm_joint_names]
        self.active_joint_ids = active_jids

        self.forces = [20.0] * self.num_active_joints
        self.jr = list(np.array(self.ul) - np.array(self.ll))
        self.jd = [0.00001] * self.num_joints
        self.jd = list(np.array(self.jd) * self.gs)
        # restposes for null space
        self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]

        self.flange_angles = [0, -math.pi, -math.pi / 2]
        self.flange_orn = self.p.getQuaternionFromEuler(self.flange_angles)

        # gripper
        self.flange_index = self.joint_dict['J6']
        self.gripper_index = self.joint_dict['gripper_to_arm']
        self.active_gripper_index = self.active_joint_dict['gripper_to_arm']

        self.initial_endeffector_angle = 0  # math.pi / 2

        self.rot_limits = [(-np.radians(175)) * 0.9, (np.radians(175)) * 0.9]

        # create a constraint to keep the fingers centered
        c = p.createConstraint(self.robot_uid,
                               8,
                               self.robot_uid,
                               9,
                               jointType=p.JOINT_GEAR,
                               jointAxis=[1, 0, 0],
                               parentFramePosition=[0, 0, 0],
                               childFramePosition=[0, 0, 0],
                               physicsClientId=self.cid)
        p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50, physicsClientId=self.cid)

        self.reset_pose()

    def get_tcp_pos(self):
        """
        get TCP currrent position in world coords for proprioceptive state

        Returns:
            gripper_pos: shape (3,)

        """
        flange_ls = self.p.getLinkState(self.robot_uid, self.flange_index, physicsClientId=self.cid)
        flange = state2matrix(flange_ls)

        # find by T_gripper @ np.linalg.inv(T_flange)

        res = (self.gripper.T_tcp_flange @ flange)[:3, 3]
        return res

        # This was the old way of computing it, can probably go
        # Extend along direction joining gripper and flange.
        # flange_ls = self.p.getLinkState(self.robot_uid, self.flange_index, physicsClientId=self.cid)
        # gripper_ls = self.p.getLinkState(self.robot_uid, self.gripper_index, physicsClientId=self.cid)
        # wrist_pos = np.array(flange_ls[0])
        # gripper_pos = np.array(gripper_ls[0])
        # extension_dir = gripper_pos - wrist_pos
        # extension_dir = extension_dir / np.linalg.norm(extension_dir)
        # extension_dir *= 0.123  # longer fingers
        # res = gripper_pos + extension_dir
        # return res

    def reset(self):
        super().reset()
        # robot base height
        pos = self.params.robot_base
        orn = self.base_orientation
        self.p.resetBasePositionAndOrientation(self.robot_uid, pos, orn, physicsClientId=self.cid)
        # adjust workspace limits
        self.workspace = self.workspace_offset + self.params.table_pos


class KukaBolt(Kuka):
    """
    KUKA with WSG50 gripper
    """

    def __init__(self, cid, *args, env_params=None, camera_pos='', **kwargs):
        super().__init__(cid, *args, env_params=env_params, camera_pos=camera_pos,
                         **kwargs)
        # first load files
        # TODO: add camera link from pybullet
        # if camera_pos == "old_mount":
        #     self.robot_path = rpath(
        #         "kuka_iiwa/kuka_lightweight_weiss_dualpivot.sdf")
        # else:
        self.robot_path = rpath(
            "kuka_iiwa/kuka_lightweight_weiss_new_cam.sdf")
        self.gripper = WSG50(cid, self, env_params, **kwargs)


class KukaSuction(Kuka):
    """
    KUKA with suction gripper.
    """

    def __init__(self, *args, env_params=None, camera_pos='', **kwargs):
        super().__init__(*args, env_params=env_params, camera_pos=camera_pos,
                         **kwargs)
        assert camera_pos == "new_mount"
        self.robot_path = rpath("kuka_iiwa/kuka_lightweight_suction.sdf")
        self.gripper = SuctionGripper(env_params)
        # suction gripper variables
        self.constraint_id = set()
        self.block_counter = 0

    @staticmethod
    def relative_pose_ba(a_trn, a_orn, b_trn, b_orn):
        '''returns the realtive pose b->a'''
        ba_trn = np.array(a_trn) - np.array(b_trn)
        ba_orn = R.from_quat(b_orn).inv() * R.from_quat(a_orn)
        return (ba_trn, ba_orn.as_quat())

    def contact_callback(self, env, object_uid, contact, action):
        '''can this be moved to robot_vacuum'''
        gripper_action = env.robot.gripper.get_state(action[-1])
        # On open action, give a cooldown for change to act
        if gripper_action == "open":
            for constraint_id in self.constraint_id:
                env.p.removeConstraint(constraint_id, physicsClientId=self.cid)
            # reset
            self.connected = set()
            self.constraint_id = set()
            self.block_counter = 10

        # quit early conditions
        if object_uid in self.connected:
            return

        if self.block_counter > 0:
            self.block_counter -= 1
            return

        robot_tip_link = 9
        robot_uid = env.robot.robot_uid
        body_a_uid, body_b_uid = contact[1], contact[2]
        link_index_a, link_index_b = contact[3], contact[4]
        # positive for separation, negative for penetration
        # contact_distance = contact[8]

        # This order seems to be consistent, but double check
        # A == object == child, B = robot == parent
        assert body_a_uid == object_uid
        assert link_index_a == -1
        assert body_b_uid == robot_uid
        assert link_index_b == robot_tip_link
        parent_body_uid = robot_uid
        parent_link_uid = robot_tip_link
        child_body_uid = object_uid
        child_link_uid = -1

        # get COMs in world coordinates
        parent_com, parent_orn = env.p.getLinkState(robot_uid,
                                                    robot_tip_link, physicsClientId=self.cid)[0:2]
        # I don't really like this API choice
        if child_link_uid == -1:
            child_com, child_orn = \
                env.p.getBasePositionAndOrientation(object_uid, physicsClientId=self.cid)
        else:
            child_com, child_orn = env.p.getLinkState(object_uid,
                                                      child_link_uid, physicsClientId=self.cid)[0:2]

        # for prinint relative poses, for trajectories
        # flangeCOM, flangeOrn = env.p.getLinkState(robot_uid,
        #                                           env.robot.flange_index)[0:2]
        # C2F = self.relative_pose_BA(flangeCOM, flangeOrn, child_com,
        #                             child_orn)

        # get collision points in COM coordinates
        pos_on_child = contact[5]  # in world coordinates
        pos_on_parent = contact[6]  # in world coordinates
        # move the contract points so that we don't lock in a collision
        # helps with stability.
        pen = np.array(pos_on_parent) - pos_on_child
        pos_on_parent += pen

        # in A COM coord
        pos_on_child_com = -np.array(child_com) + pos_on_child
        # in B COM coord
        pos_on_parent_com = -np.array(parent_com) + pos_on_parent
        # invert orientations, constraint is in world orientation

        p_orn = R.from_quat(parent_orn)
        c_orn = R.from_quat(child_orn)
        constraint_id = env.p.createConstraint(parent_body_uid, parent_link_uid,
                                               child_body_uid, child_link_uid,
                                               env.p.JOINT_FIXED,
                                               pos_on_child_com,  # jnt ax in cld lnk fr
                                               pos_on_parent_com,  # parentFramePosition
                                               pos_on_child_com,  # childFramePosition
                                               p_orn.inv().as_quat(),
                                               c_orn.inv().as_quat(),
                                               physicsClientId=self.cid)

        p.changeConstraint(constraint_id, maxForce=100, physicsClientId=self.cid)
        # save constraint
        self.connected.add(object_uid)
        self.constraint_id.add(constraint_id)
