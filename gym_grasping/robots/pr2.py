"""
PR2 Robot.
"""
import math
import numpy as np
import pybullet as p
from gym_grasping.robots.grippers_pr2 import PR2Gripper
from gym_grasping.robots.models import get_robot_path as rpath
from gym_grasping.robots.robots import Robot


class PR2(Robot):
    """
    PR2 Robot.
    """
    def __init__(self, np_random, env_sampler, act_type='continuous',
                 act_dv=.01, act_drot=0.01, joint_vel=10, gripper_rot_vel=10,
                 max_rot_diff=0.2, gripper_speed=10, gs=1,
                 use_inverse_kinematics=True):
        super().__init__(np_random, env_sampler, act_type, act_dv, act_drot,
                         joint_vel, gripper_rot_vel, max_rot_diff,
                         gripper_speed, gs, use_inverse_kinematics)

    def load(self):
        '''load the robot'''
        joint_names = ['torso_lift_joint',
                       'r_shoulder_pan_joint', 'r_shoulder_lift_joint',
                       'r_upper_arm_roll_joint', 'r_elbow_flex_joint',
                       'r_forearm_roll_joint', 'r_wrist_flex_joint',
                       'r_wrist_roll_joint', 'r_wrist_roll_joint2',
                       'r_gripper_palm_joint', 'r_gripper_l_finger_joint',
                       'r_gripper_l_finger_tip_joint',
                       'r_gripper_r_finger_joint',
                       'r_gripper_r_finger_tip_joint']
        self.base_position = [-0.2, 0.2, -1]
        self.base_orientation = p.getQuaternionFromEuler([0, 0, 0])
        # XXX currently in world cooridinates
        # robot workspace
        # self.workspace = ( [0.3, -0.2, 0.0],  # xyz low
        #                    [0.6,  0.2, 0.5])  # xyz high
        self.workspace = np.array([[0.2, -0.2, -0.177],  # xyz low
                                   [0.5, 0.2, 0.3]])  # xyz high
        # load_urdf
        super().load(joint_names)

        arm_joint_names = ['r_shoulder_pan_joint', 'r_shoulder_lift_joint',
                           'r_upper_arm_roll_joint', 'r_elbow_flex_joint',
                           'r_forearm_roll_joint', 'r_wrist_flex_joint',
                           'r_wrist_roll_joint']
        self.all_joint_ids = [self.joint_dict[jn] for jn in arm_joint_names]
        self.active_joint_ids = [self.active_joint_dict[jn] for jn in arm_joint_names]

        self.forces = [50] * self.num_active_joints
        self.forces[self.active_joint_dict['torso_lift_joint']] = 2000
        self.forces[self.active_joint_dict['r_wrist_roll_joint2']] = 20
        # joint ranges for null space
        self.jr = list(np.array(self.ul) - np.array(self.ll))
        self.jd = [0.00001] * self.num_joints
        self.jd = list(np.array(self.jd) * self.gs)

        self.flange_orn = p.getQuaternionFromEuler([0, math.pi / 2, 0])

        # gripper
        self.flange_index = self.joint_dict['r_wrist_roll_joint']
        self.gripper_index = self.joint_dict['r_wrist_roll_joint2']
        self.active_gripper_index = self.active_joint_dict['r_wrist_roll_joint2']
        self.gripper_offset = np.array([0, 0, .2]) * self.gs
        self.initial_endeffector_angle = 0

        # pose to set when reset_pose is called
        self.reset_pose()

    def randomize_configuration(self, task_info):
        '''randomize robot configuration'''
        pass

    def get_tcp_pos(self):
        '''get TCP pose'''
        wrist_pos = np.array(p.getLinkState(self.robot_uid, self.all_joint_ids[-1])[0])
        finger_midpoint_pos = \
            np.array(p.getLinkState(self.robot_uid, self.gripper.joint_ids[3])[0]) + \
            (np.array(p.getLinkState(self.robot_uid, self.gripper.joint_ids[2])[0]) -
             np.array(p.getLinkState(self.robot_uid, self.gripper.joint_ids[3])[0])) / 2
        extension_direction = finger_midpoint_pos - wrist_pos
        extension_direction = extension_direction / np.linalg.norm(extension_direction)
        extension_direction *= 0.022
        return finger_midpoint_pos + extension_direction

    def get_view_mat(self):
        '''get the view matrix'''
        camera_pos, camera_orn = p.getLinkState(self.robot_uid, self.camera_index)[:2]
        gripper_pos = self.get_tcp_pos()
        camera_eye_position = np.array(camera_pos)
        camera_target_position = np.array(gripper_pos)
        offset_lookat = np.array([0, 0, -0.1])
        camera_target_position += offset_lookat

        tmp = p.getMatrixFromQuaternion(camera_orn)
        tmp = np.array(tmp)
        tmp.shape = 3, 3

        camera_up_vector = tmp[:, 2]
        view_mat = p.computeViewMatrix(camera_eye_position,
                                       camera_target_position,
                                       camera_up_vector)
        return view_mat


class PR2Full(PR2):
    '''The full PR2 robot'''
    def __init__(self, np_random, env_sampler, act_type='continuous',
                 act_dv=.01, act_drot=0.01, joint_vel=10, gripper_rot_vel=10,
                 max_rot_diff=0.2, gripper_speed=10, act_dof='all', gs=1,
                 use_inverse_kinematics=True):
        self.gripper = PR2Gripper(joint_ids=[63, 65, 64, 66],
                                  joint_ids_active=[26, 28, 27, 29])
        self.robot_path = rpath(
            "pr2_common/pr2_description/robots/"
            "pr2_kinect_additional_joint_no_frames_reduced_arm_mass.urdf")
        # self.camera_index = 68

        self.initial_pose = [
            0.0, 0.0, 0.0, 5.413500504571876e-05,
            0.0013927545326947956, 0.0009599154519875282,
            3.1132892777285686e-05, 0.0007399053627971213,
            0.00046234862451697477, 5.456336434321085e-06,
            0.00022319043356561452, 0.00026249885948162113,
            -0.0003162880167550617, -0.004435846593336272,
            0.0003638281802773511, 0.3299999997092284, 0.0, 0.0, 0.0,
            2.9342467997295377e-08, -3.513585125457078e-07, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0890529783575378e-07, 0.0, -0.3994720631202127,
            -0.014342688672568731, -1.1924624821022791, 0.0,
            -0.9675837363553695, -1.7804026266688977, 0.0,
            -1.8884086262089044, -2.6037744032288725, 34.323384214178866,
            0.0, 0.0, 0.0, 0.0, 7.1791873602188366e-09,
            9.193695014982405e-08, 0.4999999165224206, 0.4999999346570236,
            0.5000000483388065, 0.4999999044461792, -2.063436570924603e-09,
            0, 0.0, 0.0, 2.000000054870646, 1.698607971909449e-08,
            -2.2120838426396192e-08, 0.0, 4.340507934669559e-09,
            7.767236644077147e-08, 0.0, 3.822332557414388e-09,
            2.1619912576557095e-07, 0.0, 0.0, 0.0, 0.0,
            1.7649282404608014e-08, -3.1954433891066006e-08,
            -4.055304901987335e-08, 1.0139604154084082e-06,
            1.5025081336530327e-06, 1.4497569816348444e-06,
            1.364421549557582e-08, 0.0, 0.0, 3.240592400816849e-09]
        self.initial_pose[56] = 0
        self.initial_pose[71] = 2
        self.initial_pose[72] = 1.4
        self.rp = self.initial_pose

        super().__init__(act_type, np_random, env_sampler, act_dv, joint_vel,
                         gripper_rot_vel, act_dof, gs, max_rot_diff,
                         gripper_speed, act_dof, use_inverse_kinematics)


class PR2ArmOnly(PR2):
    '''Only the arm of the PR2'''
    def __init__(self, np_random, env_sampler, act_type='continuous',
                 act_dv=.01, act_drot=0.01, joint_vel=10, gripper_rot_vel=10,
                 max_rot_diff=0.2, gripper_speed=10, act_dof='all', gs=1,
                 use_inverse_kinematics=True):
        self.gripper = PR2Gripper(joint_ids=[12, 14, 13, 15],
                                  joint_ids_active=[9, 11, 10, 12])
        self.robot_path = rpath("pr2_common/pr2_description/robots/pr2_arm_only.urdf")
        # self.camera_index = 68

        self.initial_pose = [
            0.25, -0.39947206, -0.01434269, -1.19246248, 0,
            -0.96758374, -1.78040263, 0, -1.88840863, -2.6037744, 0, 0, 0,
            0, 0, 0, 0]
        self.rp = self.initial_pose

        super().__init__(np_random, env_sampler, act_type, act_dv, act_drot,
                         joint_vel, gripper_rot_vel, max_rot_diff, gripper_speed,
                         act_dof, gs, use_inverse_kinematics)
