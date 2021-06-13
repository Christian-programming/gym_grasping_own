"""
Task manipulating SICK production objects.
"""
from math import pi
from collections import namedtuple
import numpy as np
from matplotlib import cm
from scipy.spatial.transform import Rotation as R
from gym_grasping.tasks.block.block_grasping import BlockGraspingTask
from gym_grasping.tasks.utils import pose2tuple, hsv2rgb


class SICKCADTask(BlockGraspingTask):
    """
    Task manipulating SICK production objects.
    """

    def __init__(self, cid, np_random, p, env_params, **kwargs):
        super().__init__(cid, np_random, p, env_params, control="absolute", **kwargs)

        self.trajectory = None
        self.stage = 0

        # redefine this in subclasses
        self.surface_file = "block/models/tables/table_white.urdf"

        # can be: single, combine, vacuum, window, all
        self.mode = "single"

        if self.mode == "single":
            self.object_files = ["/home/argusm/models/CAD/SICK/WL24-color.urdf", ]
            self.object_num = 1

        elif self.mode == "window":
            self.object_files = []
            self.object_position = []
            self.object_orientation = []

            base_window_pos = np.array((-.1, -0.50, 0.12))
            window_orn = (0, -pi / 2, pi/2)

            dx_base = 1/12
            base_base_pos = np.array((-dx_base, -0.65, 0.15))
            base_orn = (0, -pi/2, -pi)

            for i in range(3):
                cur_pos = base_window_pos + (i/10., 0, 0)
                self.object_files.append("/home/argusm/models/CAD/SICK/WL24_filter.urdf")
                self.object_position.append(cur_pos)
                self.object_orientation.append(window_orn)

                cur_pos = base_base_pos + (i*dx_base, 0, 0)
                self.object_files.append("/home/argusm/models/CAD/SICK/WL24_nofilter.urdf")
                self.object_position.append(cur_pos)
                self.object_orientation.append(base_orn)

        elif self.mode == "vacuum":
            self.object_files = [
                "/home/argusm/models/CAD/SICK/WL24-color.urdf",
                # "/home/argusm/models/CAD/SICK/WL24_box_inner.urdf",
                "/home/argusm/models/CAD/SICK/WL24-color.urdf",
                "/home/argusm/models/CAD/SICK/WL24-color.urdf",
                "/home/argusm/models/CAD/SICK/WL24-color.urdf",
            ]
            self.object_position = [
                (-.1, -0.52, 0.13),
                (-.1, -0.65, 0.13),
                (-0, -0.52, 0.13),
                (.1, -0.52, 0.13)
            ]
            self.object_orientation = [
                (0, -pi, -pi),
                (pi/2, 0, pi/2),
                (0, -pi, -pi),
                (0, -pi, -pi)
            ]

        elif self.mode == "all":
            self.object_files = [
                "/home/argusm/models/CAD/SICK/WL24_base.urdf",
                "/home/argusm/models/CAD/SICK/WL24_hood.urdf",
                "/home/argusm/models/CAD/SICK/WL24_filter.urdf",
                "/home/argusm/models/CAD/SICK/WL24.urdf",
                "/home/argusm/models/CAD/SICK/LMS1.urdf"
            ]
            self.object_position = [
                (0, -0.555, 0.13),
                (.1, -0.555, 0.13),
                (.2, -0.555, 0.12),
                (-.1, -0.555, 0.13),
                (.3, -0.75, 0.16),
            ]
            self.object_orientation = [
                (0, 0, -pi),
                (0, 0, -pi),
                (0, -pi/2, -pi),
                (0, -pi, -pi),
                (0, -pi/2, pi/2),
            ]
            self.object_num = 5
        else:
            print("Unknown task mode")
            raise ValueError

        self.object_num = len(self.object_files)

        # take step in case objects touches robot
        self.robot_clear_step = np.array([0, 0, .3])
        # in world coords, assumes robot at origin
        # TODO(max): somehow this is defined for flange, it should  be TCP
        self.robot_workspace_offset = [[-0.25, -0.1, 0.00],
                                       [0.25, 0.21, 0.465]]
        # generic stuff below
        self.np_random = np_random
        self.p = p
        self.surfaces = []
        self.objects = []

        self.params = env_params
        self.params.add_variable("table_size", 1, tag="geom")
        self.params.add_variable("table_pos", tag="geom", center=(0, -0.6, 0.117),
                                 d=(0, 0, .001))
        self.params.add_variable("table_orn", tag="geom", center=(0, 0, 0, 1),
                                 d=(0, 0, 0, 0))
        self.params.add_variable("object_size", tag="geom", center=1, d=0.03)
        self.params.add_variable("object_pose", tag="geom",
                                 center=(0, -0.555, 0.12, 0),
                                 d=(.2, .07, 0, 2*pi), f=pose2tuple)
        self.params.add_variable("object_to_gripper", tag="geom", center=(0, 0, 0.28),
                                 d=(0, 0, 0))
        self.params.add_variable("block_red", tag="vis", center=(.0, .7, .805),
                                 d=(.04, .04, .04), f=hsv2rgb)
        self.params.add_variable("block_blue", tag="vis", center=(.57, .49, .375),
                                 d=(.04, .04, .04), f=hsv2rgb)

    def compute_trajectory(self, env, object_id, tray_id):
        """
        Turn object relative waypoints into fixed global waypoints
        """
        assert env.robot.control == "absolute"

        if self.mode == "vacuum":
            c2f_orn = (0, 0, 0, 1)
            c2f_trn = (0, 0, 0.088)
            c2f_trn2 = (-0.01, 0, 0.12)
            delta_up = np.array((0, 0, 0.12))
        elif self.mode == "window":
            c2f_orn = (0, 0, 0, 1)
            c2f_trn = (0, 0, 0.075)
            c2f_trn2 = (0.00, -.01, 0.11)
            delta_up = np.array((0, 0, 0.12))
        else:
            raise ValueError
        Waypoint = namedtuple('Waypoint', 'anchor_id trn orn',
                              defaults=[0, c2f_trn, c2f_orn])

        wp1 = Waypoint(anchor_id=object_id, trn=c2f_trn)
        wp2 = Waypoint(anchor_id=object_id, trn=c2f_trn + delta_up)
        wp3 = Waypoint(anchor_id=tray_id, trn=c2f_trn2 + delta_up)
        wp4 = Waypoint(anchor_id=tray_id, trn=c2f_trn2)
        wp5 = Waypoint(anchor_id=tray_id, trn=c2f_trn2)
        wp6 = Waypoint(anchor_id=tray_id, trn=c2f_trn2 + delta_up)
        trajectory = [wp1, wp2, wp3, wp4, wp5, wp6]

        # bake to flange coordinates
        trajectory_f = []
        for waypoint in trajectory:
            uid = waypoint.anchor_id
            anchor_trn, anchor_orn = env.p.getBasePositionAndOrientation(uid, physicsClientId=self.cid)
            f_trn = np.array(anchor_trn) + waypoint.trn
            f_orn = R.from_quat(waypoint.orn) * R.from_quat(anchor_orn)
            wp_f = Waypoint(anchor_id=env.robot.flange_index, trn=f_trn,
                            orn=f_orn)
            trajectory_f.append(wp_f)
        self.trajectory = trajectory_f

        # plot trajectory with debug lines
        plot = False
        if plot:
            anchor_trn, anchor_orn = env.p.getBasePositionAndOrientation(object_id, physicsClientId=self.cid)
            flange_start = env.p.getLinkState(env.robot.robot_uid,
                                              env.robot.flange_index, physicsClientId=self.cid)[0]
            # env.p.addUserDebugLine([0,0,0], flange_start,
            #                        lineColorRGB=[1,0,0], lineWidth=2, physicsClientId=self.cid)
            # green to object
            # env.p.addUserDebugLine([0,0,0], anchor_trn, lineColorRGB=[0,1,0],
            #                        lineWidth=2, physicsClientId=self.cid)
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

    def policy(self, env, action):
        """
        A simple pick and place policy
        ouput:
            actions for env
        """
        env.p.resetDebugVisualizerCamera(1.3, -60, -30, [0.52, -0.2, -0.33], physicsClientId=self.cid)
        if self.trajectory is None:
            self.compute_trajectory(env, self.objects[0], self.objects[1])

        if self.stage == 0:
            if env.robot.connected:
                self.stage = 1

        elif self.stage >= 1:
            goal_xyz = self.trajectory[self.stage].trn
            cur_xyz = env.p.getLinkState(env.robot.robot_uid,
                                         env.robot.flange_index, physicsClientId=self.cid)[0]
            l_2 = np.linalg.norm(np.array(goal_xyz) - cur_xyz)
            if l_2 < 1e-3 and self.stage < len(self.trajectory)-1:
                self.stage += 1

        g_close = env.robot.gripper.close
        action = self.trajectory[self.stage].trn.tolist() + [0, g_close]
        if self.stage > 0:
            action[3] = pi/2.
        if self.stage >= 4:
            action[-1] = env.robot.gripper.open

        print("stage", self.stage, "action", action)
        return action
