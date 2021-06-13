"""
Base class for robots, this handles making robots move correctly with
various control spaces.
"""

import math
import itertools as it
import numpy as np
from gym import spaces

# TODO(max): robot control STRANGE_DELTA_Z
# This is a bit concerining, but I'm not sure where this comes from
# for the iterative control this prevents the robot from dipping down
# after initial reset, for absolute control it prevents a permanent offset
# This is not caused by gravity.
# I should write a quick test program and ask.
STRANGE_DELTA_Z = 1.99407219e-02  # this was 0.02
ZERO_ACTION = np.array([0, 0, 0, 0, 1])


class Robot:
    """
    This class manages a robot in the bullet physics simulation. It handles all
    of the IK/motion dof for the robot. This class has become a bit complicated
    because at we used it to control the model of a PR2 robot, which has
    unactuated joints.

    Specifying robot motion can be done in several ways, here is a quick
    summary. Not all permutations are required, common ones are:

    Training:
        encoding=continous, scaling=clip_dv
        space=cartesian
        frame=TCP
        mode=relative

    Manual Scripting:
        encoding=continous, scaling=raw
        space=cartesian
        frame=word
        mode=absolute

    .
    ├── DoF (degrees of freedom: 5, 6, 8, includes gripper)
    ├── encoding (action space)
    │   ├── continous
    │   │   └── scaling (scaling of inputs )
    │   │       ├── clip_dv [-1, 1] * (dv, pi)
    │   │       ├── raw
    │   │       └── tanh_dv
    │   ├── discrete
    │   └── multi-discrete
    └── space
        ├── cartesian (use_inverse_kinematics = True)
        │   ├── frame
        │   │   ├── flange
        │   │   ├── TCP (self.navigate_in_cam_frame)
        │   │   └── world
        │   └── mode (give the absolute position or deltas)
        │       ├── absolute
        │       ├── cursor
        │       └── relative
        └── joint

    TODO(max): position commands move the robot flange, this should probably be
    changed to TCP, but I'll do that later as it requires a bit more code.
    """

    # TODO(max): currently camera_pos='new_mount` controls both which robot is
    # loaded and an angle offset for the control. The robot loading should be
    # moved to the iiwa class and the angle offset should be given as numeric
    # variables
    def __init__(self, cid, p, env_params=None, act_type='continuous',
                 camera_pos="new_mount", control='relative',
                 orientation_dof="roll",
                 show_workspace=False,
                 use_inverse_kinematics=True, gs=1, **kwargs):
        self.p = p
        self.cid = cid
        assert env_params is not None
        self.params = env_params
        self.robot_path = ""
        self.initial_pose = None
        self.gripper = None
        #
        # Control Configuration
        # use_inverse_kinematics == True means cartesian control
        self.use_inverse_kinematics = use_inverse_kinematics
        self.action_delay = 0
        self.action_queue = None
        self.show_workspace = show_workspace
        #
        # Variables (those that are reset by sampler)
        self.dv = None  # env_params
        self.drot = None  # env_params
        self.joint_vel = None
        self.max_rot_diff = 0
        # Variables (other)
        self.use_simulation = True
        self.use_null_space = False
        self.use_orientation = True
        self.control = control  # relative or absolute control
        self.orientation_dof = orientation_dof
        self.navigate_in_cam_frame = True
        self.discrete_control = None
        self._cursor_control = True  # add movements to ee_pos not obs pos
        self._action_set = None
        self.action_space = None
        self.flange_angles = None
        self.flange_index = None
        self.flange_orn = None
        self.robot_uid = None
        self.num_joints = None
        self.num_active_joints = None
        self.active_gripper_index = None
        self.gripper_index = None
        self.camera_pos = camera_pos
        self.camera_index = None
        self.workspace = None
        self.workspace_offset = None
        self.rot_limits = None
        self.desired_ee_pos = None
        self.joint_positions = None
        self.desired_ee_angle = None
        self.initial_endeffector_angle = None
        self.actuated_joint_mask = []
        self.ll = []
        self.ul = []
        self.jr = []
        self.jd = []
        self.rp = []
        self.init_control(act_type)
        self.joint_dict = {}  # {joint_name: all joint ids}
        self.active_joint_dict = {}  # {joint_name: active joint ids}
        self.all_joint_ids = None
        self.active_joint_ids = None
        self.forces = None
        self.gripper_offset = None
        self.default_gripper_offset = None
        self.base_position = None
        self.base_orientation = None
        self.gs = gs

    def contact_callback(self, env, object_uid, contact, action):
        """
        callback for when a collision happens, used to implement suction
        grasping
        """

    def init_control(self, act_type):
        """
        initalize the control, can be called after setup.
        Note: the apply_action function has some flexibility, can parse formats
        too.
        """
        if not self.use_inverse_kinematics:
            act_dof = 8
            ctrl_nms = ['J0', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'grip']
        elif self.use_inverse_kinematics:
            if self.orientation_dof == "roll":
                act_dof = 5
                ctrl_nms = ["posX", "posY", "posZ", "roll", "grip"]
            elif self.orientation_dof == "roll pitch":
                act_dof = 6
                ctrl_nms = ["posX", "posY", "posZ", "roll", "pitch", "grip"]
                raise NotImplementedError
            elif self.orientation_dof == "roll pitch yaw":
                act_dof = 7
                ctrl_nms = ["posX", "posY", "posZ", "roll", "pitch", "yaw",
                            "grip"]
            elif self.orientation_dof == "quaternion":
                act_dof = 8
                ctrl_nms = ["posX", "posY", "posZ", "q_x", "q_z", "q_z", "w",
                            "grip"]
            else:
                raise ValueError(f"Orientation dof: {self.orientation_dof}")

        # this is the DoF of the underlying action, i.e. if we have discrete
        # control it is not set to 1 but the DoF for the robot
        self.act_dof = act_dof
        self.control_names = ctrl_nms

        if act_type == 'discrete':
            self.discrete_control = True
            if not self.use_inverse_kinematics:
                raise NotImplementedError
            delta_v = 1
            self._action_set = tuple(it.product([-delta_v, 0, delta_v],
                                     repeat=act_dof))
            self.action_space = spaces.Discrete(len(self._action_set))
        elif act_type == 'multi-discrete':
            self.discrete_control = True

            class ActionConversion:
                """Mini-class for dict lookup API"""

                def __getitem__(self, discrete_action):
                    cont_action = np.array(discrete_action) - 1
                    # gripper action is only binary, robot actions have three components
                    assert discrete_action[-1] < 2
                    cont_action[-1] = -1 if discrete_action[-1] == 0 else 1
                    return cont_action
            self._action_set = ActionConversion()
            self.action_space = spaces.MultiDiscrete([3, ] * (act_dof-1) + [2, ])
        elif act_type == 'continuous':
            self.discrete_control = False
            action_high = np.ones(act_dof)
            self.action_space = spaces.Box(-action_high, action_high,
                                           dtype=np.float32)
        else:
            raise ValueError(
                "Unknown act type:" + str(act_type) +
                "must be: 'discrete', 'multi-discrete', 'continuous'")

    # this was named accurateCalculateInverseKinematics
    def accurate_ik(self, target_pos, target_orn=None, threshold=.00005,
                    max_iter=500):
        """
        compute the inverse kinematics
        Args:
            target_pos in word coordinates
        Returns:
            joint poses: [7xfloat] for kuka
        """
        joint_poses = None
        for _ in range(max_iter):
            joint_poses = self.p.calculateInverseKinematics(
                self.robot_uid,
                self.flange_index,
                target_pos,
                target_orn,
                physicsClientId=self.cid)
            #                   , lowerLimits=self.ll,
            #                   upperLimits=self.ul,
            #                   jointRanges=self.jr,
            #                   restPoses=self.rp)

            for i, j in zip(self.active_joint_ids, self.all_joint_ids):
                self.p.resetJointState(self.robot_uid, j, joint_poses[i], physicsClientId=self.cid)
            new_pos = self.p.getLinkState(self.robot_uid, self.flange_index, physicsClientId=self.cid)[0]
            diff = [target_pos[0] - new_pos[0],
                    target_pos[1] - new_pos[1],
                    target_pos[2] - new_pos[2]]
            dist2 = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
            if dist2 < threshold:
                return joint_poses
        # print ("Num iter: "+str(iter) + "threshold: "+str(dist2))
        return joint_poses

    def load(self, joint_names):
        """
        load the robot file and set all relevant variables.
        """
        # visual workspace limits
        if self.show_workspace:
            half_extents = (self.workspace[1] - self.workspace[0]) / 2
            pos = self.workspace[0] + half_extents
            ws_limit_id = self.p.createVisualShape(shapeType=self.p.GEOM_BOX,
                                                   rgbaColor=[0, 1, 0, 0.5],
                                                   halfExtents=half_extents,
                                                   visualFramePosition=pos,
                                                   physicsClientId=self.cid)
            self.p.createMultiBody(baseVisualShapeIndex=ws_limit_id,
                                   physicsClientId=self.cid)

        # load the robot
        if self.robot_path.endswith(".urdf"):
            base_orn = self.base_orientation
            self.robot_uid = self.p.loadURDF(self.robot_path,
                                             globalScaling=self.gs,
                                             basePosition=self.base_position,
                                             baseOrientation=base_orn,
                                             physicsClientId=self.cid)
        else:
            objects = self.p.loadSDF(self.robot_path, globalScaling=self.gs,
                                     physicsClientId=self.cid)
            self.robot_uid = objects[0]
            self.base_position = self.params.robot_base
            self.p.resetBasePositionAndOrientation(self.robot_uid,
                                                   self.base_position,
                                                   self.base_orientation,
                                                   physicsClientId=self.cid)

        joint_names = set(joint_names)
        self.num_joints = self.p.getNumJoints(self.robot_uid, physicsClientId=self.cid)
        self.actuated_joint_mask = []
        self.num_active_joints = 0
        self.ll = []
        self.ul = []
        for i in range(self.num_joints):
            j = self.p.getJointInfo(self.robot_uid, i, physicsClientId=self.cid)
            joint_name = str(j[1], 'utf-8')
            if joint_name in joint_names:
                self.joint_dict[joint_name] = j[0]
                self.active_joint_dict[joint_name] = self.num_active_joints
            elif joint_name == 'camera_joint':
                self.camera_index = j[0]
            # else ignore joint
            if j[2] is not self.p.JOINT_FIXED:
                self.num_active_joints += 1
                self.actuated_joint_mask.append(i)
            self.ll.append(j[8])
            self.ul.append(j[9])

    def reset_pose(self, target_pos_orn=None, angle=None, gripper_pose=1):
        """
        factorized code for resetting position, can be called separately in
        theory
        """
        joint_positions = self.initial_pose
        assert (len(joint_positions) == self.num_joints), \
            "joint_positions was {} should be {}".format(
                len(joint_positions), self.num_joints)
        for joint_index in range(self.num_joints):
            self.p.resetJointState(self.robot_uid, joint_index,
                                   joint_positions[joint_index],
                                   physicsClientId=self.cid)
        if target_pos_orn is not None:
            joint_positions = self.get_task_dependent_pose(target_pos_orn)

        # reset robot
        if self.use_inverse_kinematics:
            flange_ls = self.p.getLinkState(self.robot_uid, self.flange_index,
                                            physicsClientId=self.cid)
            self.desired_ee_pos = list(flange_ls[0])
            self.desired_ee_pos[2] += STRANGE_DELTA_Z
        else:
            joint_positions = np.array(joint_positions)[self.all_joint_ids]
            self.joint_positions = joint_positions
        # reset end effector orientation
        if angle is None:
            pre_clip_angle = self.initial_endeffector_angle
        else:
            pre_clip_angle = angle
        self.desired_ee_angle = np.clip(pre_clip_angle,
                                        self.rot_limits[0], self.rot_limits[1])
        self.p.resetJointState(self.robot_uid, self.active_gripper_index,
                               self.desired_ee_angle, physicsClientId=self.cid)
        # reset gripper pose
        # reset needs to be called one, check if this is the right place
        self.gripper.reset()
        self.gripper.reset_pose(gripper_pose)

    def reset(self):
        """
        reset dynamics variables, that may be changed by param randomization
        """
        # updates self.gripper.speed
        self.gripper.reset()
        if self.action_delay:
            default_action = np.array([0, 0, 0, 0, 1])
            a_q = [default_action for _ in range(len(self.action_queue))]
            self.action_queue = a_q
        f_s = self.params.frameskip
        self.dv = self.params.robot_dv * 4.0 / f_s
        self.drot = self.params.robot_drot * 4.0 / f_s
        self.joint_vel = self.params.joint_vel
        self.max_rot_diff = self.params.max_rot_diff

    def get_observation(self):
        """
        returns proprioceptive state as observation
        """
        gripper_pos = self.get_tcp_pos()
        grip_ls = self.p.getLinkState(self.robot_uid, self.gripper_index, physicsClientId=self.cid)[1]
        gripper_orn = self.p.getEulerFromQuaternion(grip_ls)[2]
        # joint states of 7 arm joints & gripper finger angle
        arm_joint_states = []
        for i in [*self.all_joint_ids[:-1], self.active_gripper_index]:
            arm_joint_states.append(self.p.getJointState(self.robot_uid, i, physicsClientId=self.cid)[0])

        gripper_opening_width = self.gripper.get_opening_width()
        observation = np.array([*gripper_pos, gripper_orn, *arm_joint_states,
                                gripper_opening_width])
        return observation

    def do_it_control(self, dxyz, rot_action):
        """
        do iterative dof, give out target position in world coords.

        Args:
            dxyz: delta xyz as (3,)
            rot_action: rotation as scalar
        Returns:
            pre_clip: xyz in world coordinates
            pre_clip_angle: angle in world coordinates
        """
        if self.dv:
            dxyz = dxyz * self.dv

        if self.navigate_in_cam_frame:
            # TODO(max): this needs to be simplified, don't need to
            # quat->euler->matrix
            grip_ls = self.p.getLinkState(self.robot_uid,
                                          self.gripper_index, physicsClientId=self.cid)[1]
            gripper_orn = self.p.getEulerFromQuaternion(grip_ls)[2]
            if self.camera_pos == "new_mount":
                gripper_orn += math.pi
            rot_mat = np.array([[np.cos(gripper_orn), -np.sin(gripper_orn), 0],
                                [np.sin(gripper_orn), np.cos(gripper_orn), 0],
                                [0, 0, 1]])

            dxyz = np.matmul(rot_mat, dxyz)

        current_ee_angle = self.p.getJointState(self.robot_uid,
                                                self.active_gripper_index,
                                                physicsClientId=self.cid)[0]

        if self._cursor_control:
            # limit difference between desired EE pos and actual pos by
            # clipping dxyz
            # current_ee_pos = np.array(self.p.getLinkState(self.robot_uid, self.flange_index)[0]) + np.array([0, 0, 0.02])
            # if current_ee_pos[0] + dxyz[0] < self.desired_ee_pos[0] and dxyz[0] > 0 or \
            #         current_ee_pos[0] + dxyz[0] > self.desired_ee_pos[0] and dxyz[0] < 0:
            #     dxyz[0] = 0
            # if current_ee_pos[1] + dxyz[1] < self.desired_ee_pos[1] and dxyz[1] > 0 or \
            #         current_ee_pos[1] + dxyz[1] > self.desired_ee_pos[1] and dxyz[1] < 0:
            #     dxyz[1] = 0
            # if current_ee_pos[2] + dxyz[2] < self.desired_ee_pos[2] and dxyz[2] > 0 or \
            #         current_ee_pos[2] + dxyz[2] > self.desired_ee_pos[2] and dxyz[2] < 0:
            #     dxyz[2] = 0
            pre_clip = self.desired_ee_pos + dxyz
            # limit difference between desired EE angle and actual angle by
            # clipping da
            rot_z_diff = self.desired_ee_angle - current_ee_angle
            delta_a = rot_action * self.drot
            if rot_z_diff > self.max_rot_diff and delta_a > 0 \
               or rot_z_diff < -self.max_rot_diff and delta_a < 0:
                delta_a = 0
            pre_clip_angle = self.desired_ee_angle + delta_a

        else:
            flange_ls = self.p.getLinkState(self.robot_uid,
                                            self.flange_index,
                                            physicsClientId=self.cid)[0]
            current_ee_pos = np.array(flange_ls) + \
                np.array([0, 0, STRANGE_DELTA_Z])
            base_pos = current_ee_pos
            pre_clip = base_pos + dxyz
            pre_clip_angle = current_ee_angle + rot_action
            raise NotImplementedError

        return pre_clip, pre_clip_angle

    def get_ik_control(self, action):
        """
        Get the inverse kinematics control. Also does endEffector code even
        though its not technically IK.

        Iterface should be similar to: Gripper*:get_control

        Args:
            action: xyz + rot + gripper
        Returns:
            target_pose: joint poses for robots
        """
        xyz = np.array(action[0:3], dtype=np.float32) * self.gs
        angle = action[3]

        if self.control == "absolute":
            pre_clip, pre_clip_angle = xyz, angle
            pre_clip += np.array([0, 0, STRANGE_DELTA_Z])
        elif self.control == "relative":
            pre_clip, pre_clip_angle = self.do_it_control(xyz, angle)
        else:
            raise ValueError

        # limit motion
        self.desired_ee_pos = np.clip(pre_clip, self.workspace[0],
                                      self.workspace[1])
        self.desired_ee_angle = np.clip(pre_clip_angle, self.rot_limits[0],
                                        self.rot_limits[1])

        if self.control == "absolute":
            if np.any(self.desired_ee_pos != pre_clip):
                where_clip = pre_clip != self.desired_ee_pos
                print("Warning: workspace limit pos",
                      " ".join(np.array(["x", "y", "z"])[where_clip]), ":",
                      pre_clip[where_clip], "->",
                      self.desired_ee_pos[where_clip])

            if np.any(self.desired_ee_angle != pre_clip_angle):
                # where_clip = pre_clip_angle != self.desired_ee_angle
                # print("Warning: workspace limit angle",
                #      " ".join(np.array(["a", "b", "c"])[where_clip]), ":",
                #      pre_clip_angle[where_clip], "->",
                #      self.desired_ee_angle[where_clip])
                print("Warning: workspace limit angle",
                      "a: ", pre_clip_angle, "->",
                      self.desired_ee_angle)

        # compute robot poses
        pos = self.desired_ee_pos
        if len(action) >= 7:
            # TODO(max): hacky, see defn of flange_orn
            # [-1, 1] -> [-pi, pi]
            flange_angles_delta = np.array([action[5], 0, action[6]]) * math.pi
            orn = self.flange_angles + flange_angles_delta
            orn = self.p.getQuaternionFromEuler(orn)
        else:
            orn = self.flange_orn

        if self.use_null_space:
            if self.use_orientation:
                jnt_ps = self.p.calculateInverseKinematics(self.robot_uid,
                                                           self.flange_index,
                                                           pos, orn,
                                                           lowerLimits=self.ll,
                                                           upperLimits=self.ul,
                                                           jointRanges=self.jr,
                                                           restPoses=self.rp,
                                                           jointDamping=self.jd,
                                                           physicsClientId=self.cid
                                                           )
            else:
                jnt_ps = self.p.calculateInverseKinematics(self.robot_uid,
                                                           self.flange_index,
                                                           pos,
                                                           lowerLimits=self.ll,
                                                           upperLimits=self.ul,
                                                           jointRanges=self.jr,
                                                           restPoses=self.rp,
                                                           jointDamping=self.jd,
                                                           physicsClientId=self.cid
                                                           )
        else:
            if self.use_orientation:
                jnt_ps = self.p.calculateInverseKinematics(self.robot_uid,
                                                           self.flange_index,
                                                           pos, orn,
                                                           physicsClientId=self.cid)
            else:
                jnt_ps = self.p.calculateInverseKinematics(self.robot_uid,
                                                           self.flange_index,
                                                           pos,
                                                           physicsClientId=self.cid)
        target_pose = np.array(self.initial_pose)[self.actuated_joint_mask]
        target_pose[self.active_joint_ids] = \
            np.array(jnt_ps)[self.active_joint_ids]
        target_pose[self.active_gripper_index] = self.desired_ee_angle
        return target_pose

    def apply_action(self, action):
        """
        Apply a given action to a robot.
        """
        # delay actions if needed
        if self.action_queue:
            self.action_queue.append(action)
            action = self.action_queue.pop(0)

        if action is None:
            return

        if self.discrete_control:
            # convert discrete ac to continuous
            action = np.array(self._action_set[action])
        assert len(action) == self.act_dof, \
            f"action dof is: {len(action)}, expected {self.act_dof}, {self.control_names}"

        if np.any(np.isnan(np.array(action))):
            raise ValueError("Action contains nans", action)
            # we could be lenient here, but its proably catching bugs
            # action = np.nan_to_num(action)

        # TODO(max): do we want to use tanh activations?
        if self.control == 'relative':
            action = np.clip(action, -1, 1)

        if not self.use_inverse_kinematics:
            if not self.use_simulation:
                raise NotImplementedError
            for i, joint_id in enumerate(self.all_joint_ids):
                action = action[i]
                self.joint_positions[i] += action * self.dv
                if self.ul[joint_id] != -1:
                    self.joint_positions[i] = np.clip(self.joint_positions[i],
                                                      self.ll[joint_id],
                                                      self.ul[joint_id])
                self.p.setJointMotorControl2(self.robot_uid, joint_id,
                                             self.p.POSITION_CONTROL,
                                             targetPosition=self.joint_positions[i],
                                             force=self.forces[self.active_joint_ids[i]],
                                             physicsClientId=self.cid)

            self.p.setJointMotorControl2(self.robot_uid,
                                         self.all_joint_ids[-1] + 1,
                                         self.p.POSITION_CONTROL,
                                         targetPosition=0,
                                         force=50,
                                         physicsClientId=self.cid)
            # now do gripper
            hand_poses, hand_forces = self.gripper.get_control(action[-1])
            self.p.setJointMotorControlArray(self.robot_uid,
                                             self.gripper.joint_ids,
                                             self.p.POSITION_CONTROL,
                                             targetPositions=hand_poses,
                                             forces=hand_forces,
                                             physicsClientId=self.cid)
            return

        # in this case we use cartesian control
        poses = self.get_ik_control(action)
        forces = np.array(self.forces)
        # gripper controller
        hand_poses, hand_forces = self.gripper.get_control(action[4])
        poses[self.gripper.joint_ids] = hand_poses
        forces[self.gripper.joint_ids] = hand_forces
        velocities = np.ones_like(forces) * self.joint_vel
        velocities[self.active_gripper_index] = self.gripper.rot_vel
        if len(self.gripper.joint_ids) > 0:
            velocities[self.gripper.joint_ids[0]] = self.gripper.speed
        if len(self.gripper.joint_ids) > 1:
            velocities[self.gripper.joint_ids[1]] = 2*self.gripper.speed

        if self.use_simulation:
            # TODO(lukas): can we do the same thing with
            # setJointMotorControlArray?
            for i in self.actuated_joint_mask:
                self.p.setJointMotorControl2(bodyIndex=self.robot_uid,
                                             jointIndex=i,
                                             controlMode=self.p.POSITION_CONTROL,
                                             force=forces[i],
                                             targetPosition=poses[i],
                                             maxVelocity=velocities[i],
                                             physicsClientId=self.cid)
        else:
            # reset the joint state ignoring all dynamics
            # not recommended to use during simulation
            for i in self.actuated_joint_mask:
                self.p.resetJointState(self.robot_uid, i, poses[i],
                                       physicsClientId=self.cid)

        # self.p.removeAllUserDebugItems()
        # self.p.addUserDebugLine([0,0,0], self.p.getLinkState(self.robot_uid,
        #                         self.flange_index)[0], lineColorRGB=[1,0,0],
        #                         lineWidth=2, physicsClientId=self.cid)
        # self.p.addUserDebugLine([0,0,0], self.get_tcp_pos(),
        #                         lineColorRGB=[1,1,0], lineWidth=2, physicsClientId=self.cid)
        # self.p.addUserDebugLine([0,0,0], self.desired_ee_pos,
        #                         lineColorRGB=[0,1,1], lineWidth=2, physicsClientId=self.cid)
        # self.p.addUserDebugLine([0,0,0], self.desired_ee_pos,
        #                         lineColorRGB=[0,1,1], lineWidth=2, physicsClientId=self.cid)

    def get_task_dependent_pose(self, object_pos_orn):
        '''position relative to a task object, for reset'''
        object_pos, object_orn = object_pos_orn
        flange_pos_desired = np.array(object_pos)

        if object_orn is None:
            flange_orn_desired = np.array(self.flange_orn)
        else:
            flange_orn_desired = object_orn
        joint_poses = self.accurate_ik(flange_pos_desired, flange_orn_desired)
        full_joint_poses = np.zeros(self.num_joints)
        full_joint_poses[self.actuated_joint_mask] = joint_poses
        return list(full_joint_poses)

    def get_tcp_pos(self):
        '''for proprioceptive state this needs to be implemented by a
        derived class'''
        raise NotImplementedError

    def set_navigate_world(self):
        '''change control mode to world'''
        self.navigate_in_cam_frame = False
