"""
Task manipulating SICK production objects.
"""
from math import pi
# from collections import namedtuple
import numpy as np
from matplotlib import cm
from scipy.spatial.transform import Rotation as R
from gym_grasping.utils import state2matrix
from gym_grasping.tasks.block.block_grasping import BlockGraspingTask
from gym_grasping.tasks.utils import pose2tuple, hsv2rgb
from gym_grasping.robots.grippers import SuctionGripper


class PickNPlaceTask(BlockGraspingTask):
    """
    Task manipulating SICK production objects.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trajectory = None
        self.stage = 0

        # redefine this in subclasses
        self.surface_file = "block/models/tables/table_white.urdf"

        self.object_files = [
            "block/models/block_wood.urdf",
            "pick_and_place/models/surface_red.urdf",
            "block/models/block_wood.urdf",
            "block/models/block_wood.urdf",
        ]
        self.object_position = [
            (-.1, -0.52, 0.0),
            (-.1, -0.65, 0.0),
            (-0, -0.52, 0.0),
            (.1, -0.52, 0.0)
        ]
        self.object_orientation = [
            (0, -pi, -pi),
            (0, -pi, -pi),
            (0, -pi, -pi),
            (0, -pi, -pi)
        ]
        self.object_num = len(self.object_files)

        # take step in case objects touches robot
        self.robot_clear_step = np.array([0, 0, .3])
        # in world coords, assumes robot at origin
        # TODO(max): somehow this is defined for flange, it should  be TCP
        self.robot_workspace_offset = [[-0.25, -0.1, 0.00-.04],
                                       [0.25, 0.21, 0.465-.04]]

        # generic stuff below
        self.surfaces = []
        self.objects = []

        self.params.add_variable("table_size", 1, tag="geom")
        self.params.add_variable("table_pos", tag="geom", center=(0, -0.6, 0.117),
                                 d=(0, 0, .001))
        self.params.add_variable("table_orn", tag="geom", center=(0, 0, 0, 1),
                                 d=(0, 0, 0, 0))
        self.params.add_variable("object_size", tag="geom", center=1, d=0.03)
        self.params.add_variable("object_pose", tag="geom",
                                 center=(0, -0.555, 0.12, 0),
                                 d=(.2, .07, 0, 2*pi), f=pose2tuple)
        # this is the version for suction gripper
        # self.params.add_variable("object_to_gripper", tag="geom",
        #                         center=(0, 0, 0.28), d=(0, 0, 0))
        # this is the version for weiss gripper
        self.params.add_variable("object_to_gripper", tag="geom",
                                 center=(0, 0, 0.32), d=(0, 0, 0))
        self.params.add_variable("block_red", tag="vis", center=(.0, .7, .805),
                                 d=(.04, .04, .04), f=hsv2rgb)
        self.params.add_variable("block_blue", tag="vis", center=(.57, .49, .375),
                                 d=(.04, .04, .04), f=hsv2rgb)

    def reset(self):
        super().reset()
        #
        # trajectory policy
        self.trajectory = None
        self.stage = 0

    def step(self, env, action):
        self.ep_length += 1
        done = False
        reward = 0.
        info = {}

        success = False
        first_robot_contact = True
        # iterate over object contacts
        for i, block_uid in enumerate(self.objects):
            #print("block_uid", block_uid)
            contacts = self.p.getContactPoints(block_uid, physicsClientId=self.cid)

            for cnt in contacts:
                if first_robot_contact and cnt[2] == env.robot.robot_uid:
                    env.robot.contact_callback(env=env, object_uid=block_uid,
                                               contact=cnt, action=action)
                    first_robot_contact = False

                if block_uid == self.objects[1]:
                    if cnt[2] in self.objects:
                        success = True

            if success:
                done = True
                reward = 1.0

        return None, reward, done, info

    def compute_trajectory(self, env, object_id, tray_id):
        """
        Turn object relative waypoints into fixed global waypoints.

        Helper function for self.policy. Control is in flange coords, so we get
        there via:
            flange <- TCP <- object

        """
        assert env.robot.control == "absolute"

        c2f_orn = (0, 0, 0, 1)
        c2f_trn = (0, 0, 0.008)
        c2f_trn2 = (-0.01, 0, 0.04)
        delta_up = np.array((0, 0, 0.05))

        # TODO(max): python 3.7
        # Waypoint = namedtuple('Waypoint', 'anchor_id trn orn',
        #                      defaults=[0, c2f_trn, c2f_orn])
        class Waypoint:
            def __init__(self, anchor_id=0, trn=c2f_trn, orn=c2f_orn):
                self.anchor_id = anchor_id
                self.trn = trn
                self.orn = orn

        wp1 = Waypoint(anchor_id=object_id, trn=c2f_trn)
        wp2 = Waypoint(anchor_id=object_id, trn=c2f_trn + delta_up)
        wp3 = Waypoint(anchor_id=tray_id, trn=c2f_trn2 + delta_up)
        wp4 = Waypoint(anchor_id=tray_id, trn=c2f_trn2)
        wp5 = Waypoint(anchor_id=tray_id, trn=c2f_trn2)
        wp6 = Waypoint(anchor_id=tray_id, trn=c2f_trn2 + delta_up)
        trajectory = [wp1, wp2, wp3, wp4, wp5, wp6]

        T_flange_tcp = np.linalg.inv(env.robot.gripper.T_tcp_flange)

        # bake to flange coordinates
        trajectory_f = []
        for waypoint in trajectory:
            uid = waypoint.anchor_id
            obj_t = env.p.getBasePositionAndOrientation(uid, physicsClientId=self.cid)
            obj_T = state2matrix(obj_t)

            T_tcp_obj = state2matrix((waypoint.trn, waypoint.orn))
            T_flange = T_flange_tcp @ T_tcp_obj @ obj_T

            f_trn = T_flange[:3, 3]
            f_orn = R.from_matrix(T_flange[:3, :3])
            wp_f = Waypoint(anchor_id=env.robot.flange_index, trn=f_trn,
                            orn=f_orn)

            trajectory_f.append(wp_f)
        self.trajectory = trajectory_f

        # plot trajectory with debug lines
        plot = False
        if plot:
            anchor_trn, anchor_orn = \
                env.p.getBasePositionAndOrientation(object_id, physicsClientId=self.cid)
            flange_start = env.p.getLinkState(env.robot.robot_uid,
                                              env.robot.flange_index, physicsClientId=self.cid)[0]
            env.p.addUserDebugLine([0, 0, 0], flange_start,
                                   lineColorRGB=[1, 0, 0], lineWidth=2, physicsClientId=self.cid)
            # green to object
            env.p.addUserDebugLine([0, 0, 0], anchor_trn, lineColorRGB=[0, 1, 0],
                                   lineWidth=2, physicsClientId=self.cid)
            viridis = cm.get_cmap('viridis', len(trajectory))
            for i, cur_wp in enumerate(trajectory_f):
                if i == 0:
                    prev = flange_start
                else:
                    prev = trajectory_f[i-1].trn
                cur = cur_wp.trn
                rgb = viridis.colors[i, 0:3]
                env.p.addUserDebugLine(prev, cur, lineColorRGB=rgb,
                                       lineWidth=2, physicsClientId=self.cid)

    def policy(self, env, action=None):
        """
        A simple pick and place policy
        ouput:
            actions for env
        """
        assert action is None
        # env.p.resetDebugVisualizerCamera(1.3, -60, -30, [0.52, -0.2, -0.33], physicsClientId=self.cid)

        if self.trajectory is None:
            self.compute_trajectory(env, self.objects[0], self.objects[1])

        if self.stage == 0:
            is_suction = isinstance(env.robot.gripper, SuctionGripper)
            if is_suction:
                g_action = env.robot.gripper.close_action
                if env.robot.connected:
                    self.stage = 1

            else:
                # parallel gripper
                robot_z = env._observation_state[2]
                threshold_z = 0.138
                if robot_z > threshold_z:
                    g_action = env.robot.gripper.open_action
                    self.countdown = 3
                else:
                    g_action = env.robot.gripper.close_action

                    # countdown is a quick and dirty solution, should be
                    # be something more like if env.robot.connected
                    self.countdown -= 1

                    # if env.robot.connected:
                    if self.countdown < 0:
                        self.stage = 1

        elif self.stage >= 1:
            goal_xyz = self.trajectory[self.stage].trn
            cur_xyz = env.p.getLinkState(env.robot.robot_uid,
                                         env.robot.flange_index, physicsClientId=self.cid)[0]
            l_2 = np.linalg.norm(np.array(goal_xyz) - cur_xyz)
            if l_2 < 1e-3 and self.stage < len(self.trajectory)-1:
                self.stage += 1

            g_action = env.robot.gripper.close_action

        action = self.trajectory[self.stage].trn.tolist() + [0, g_action]

        if self.stage > 0:
            action[3] = pi/2.
        if self.stage >= 4:
            action[-1] = env.robot.gripper.open_action

        done = False
        if self.stage == 5:
            done = True

        return action, done
