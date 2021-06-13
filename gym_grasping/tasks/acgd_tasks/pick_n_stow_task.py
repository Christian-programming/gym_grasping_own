from math import pi
import numpy as np
import pybullet as p
from gym_grasping.tasks.utils import opath
from gym_grasping.tasks.utils import pose2tuple, hsv2rgb


class PickAndStowTask:

    def __init__(self, cid, np_random, bullet, env_params, *args, table_surface='white', penalty=0.5, **kwargs):
        self.cid = cid
        # table with wood texture
        if table_surface == 'cork':
            self.surface_file = "acgd_tasks/models/tables/table_texture.urdf"
        else:
            self.surface_file = "acgd_tasks/models/tables/table_primitive.urdf"
            # self.surface_file = "block/models/table_fabric/table_fabric.urdf"
        # take step in case objects touches robot
        self.robot_clear_step = np.array([0, 0, .3])
        # in world coords, assumes robot at origin
        # self.robot_workspace_offset = [[-0.25, -0.7, 0.41],
        #                         [0.25, -0.41, 0.58]]
        self.robot_workspace_offset = [[-0.25, -0.15, 0.3],
                                       [0.25, 0.15, 0.465]]
        # generic stuff below
        self.np_random = np_random
        self.p = bullet  # is this needed?
        self.surfaces = []
        self.objects = []

        self.params = env_params
        self.params = env_params
        self.params.add_variable("table_size", 1, tag="geom")
        self.params.add_variable("table_pos", tag="geom", center=(0, -0.55, 0.117), d=(0, 0, .001))
        self.params.add_variable("table_orn", tag="geom", center=(0, 0, 0, 1), d=(0, 0, 0, 0))
        self.params.add_variable("object_size", tag="geom", center=1, d=0.03)
        # self.params.add_variable("object_pose", tag="geom", center=(0.05, -0.555, 0.015, 0),
        #                          d=(.05, .05, 0, 2*pi), f=pose2tuple)
        self.params.add_variable("object_pose", tag="geom", ll=(0, -0.625, 0.015, 0),
                                 ul=(0.1, -0.475, 0.015, 2*pi), mu_s=(0.025, -0.55, 0.015, pi),
                                 mu_e=(0.035, -0.55, 0.015, pi), r_s=(0.025, 0.025, 0, pi),
                                 r_e=(0.05, 0.075, 0, pi), f=pose2tuple)

        if table_surface == 'white' or table_surface == 'cork':
            self.params.add_variable("block_blue", tag="vis", center=(.57, .49, .375), d=(.04, .04, .04), f=hsv2rgb)
            self.params.add_variable("table_green", tag="vis", center=(0.15, .1, .95), d=(.05, .05, .05), f=hsv2rgb)
            self.params.add_variable("block_red", tag="vis", center=(.011, .526, .721), d=(.04, .04, .04), f=hsv2rgb)
        else:
            raise ValueError(
                "Invalid or missing argument: table_surface. Passed argument: {}".format(str(table_surface)))

        self.params.add_variable("min_obj_dist", .03, tag="geom")
        self.params.add_variable("object_to_gripper", tag="geom", ll=(0, -0.5, 0.3), ul=(0.3, 0.5, 0.5),
                                 mu_s=(0.15, -0.025, 0.35), mu_e=(0.05, -0.025, 0.4), r_s=(0.01, 0.01, 0.01),
                                 r_e=(0.03, 0.04, 0.02))
        # multiple objectsare supposed nowe
        self.object_files = [opath('acgd_tasks/models/block_blue.urdf')]
        assert len(self.object_files), "no files found"
        self.num_objects = len(self.object_files)
        # for i in range(1, self.num_objects):
        #     self.params.add_variable("block_{}".format(i), tag="geom", center=(0, 0, 0, 0), d=(0.08, 0.065, 0, 2 * pi),
        #                              r_s=(0.05, 0.05, 0, 2 * pi), r_e=(0.08, 0.065, 0.0, 2 * pi), f=pose2tuple)
        self._cnt_obj_in_box = 0
        self._contact_penalty = 0
        self.params.add_variable("gripper_rot", center=-pi / 4, d=pi / 12)
        self.penalty_coeff = penalty

    def load_scene(self):
        """
        Called once per enviroment
        """
        self.objects = []

        # static
        table = self.p.loadURDF(opath(self.surface_file),
                                self.params.table_pos,
                                self.params.table_orn,
                                globalScaling=self.params.table_size,
                                physicsClientId=self.cid)
        self.surfaces = [table]
        # load box
        self.box_position = [-0.07, -0.53, -0.015]
        self.box_position[2] += self.params.table_pos[2]
        box_orientation = self.p.getQuaternionFromEuler([0.05, -0.03, -pi/2])
        box = p.loadURDF(fileName=opath("acgd_tasks/models/boxlidleft/box_bright.urdf"),
                         basePosition=self.box_position,
                         baseOrientation=box_orientation, physicsClientId=self.cid)

        self.surfaces.append(box)
        # don't load dynamic stuff here because needs resize

    def reset_from_curriculum(self):
        pass

    def reset(self):
        """
        Called once per episode
        """
        # episdoe specific variables
        self.ep_length = 0
        self.start_pos = []

        # sample table height offset
        table_pos = self.params.table_pos
        self.p.resetBasePositionAndOrientation(self.surfaces[0], table_pos,
                                               p.getQuaternionFromEuler([0, 0, pi / 2]),
                                               physicsClientId=self.cid)
        self.box_position = [-0.07, -0.53, -0.015]
        self.box_position[2] += self.params.table_pos[2]
        box_orientation = self.p.getQuaternionFromEuler([0.05, -0.03, -pi / 2])
        self.p.resetBasePositionAndOrientation(self.surfaces[1], self.box_position,
                                               box_orientation, physicsClientId=self.cid)

        # can't do the same for objects here because we want to vary size
        self.load_episode()
        self._cnt_obj_in_box = 0
        self._contact_penalty = 0
        self._change_object_colors()

    def sample_object_pose(self):
        objs_center_pos, orn = self.params.object_pose
        objs_center_pos[2] += self.params.table_pos[2]
        pos_objs, orn_objs = [objs_center_pos], [orn]
        return pos_objs, orn_objs

    def load_episode(self):
        if len(self.objects):
            for uid in self.objects:
                p.removeBody(uid, physicsClientId=self.cid)
            self.objects = []
        globalScaling = 1
        pos_objs, orn_objs = self.sample_object_pose()

        for mdl_file, pos, orn in zip(self.object_files, pos_objs, orn_objs):
            block_uid = p.loadURDF(mdl_file, pos, orn,
                                   globalScaling=globalScaling, physicsClientId=self.cid)
            self.objects.append(block_uid)
            self.start_pos.append((pos, orn))
            p.changeDynamics(block_uid, -1, restitution=self.params.restitution, physicsClientId=self.cid)

    def _change_object_colors(self):
        blue_color = self.params.block_blue
        table_color = self.params.table_green

        if table_color is not None:
            p.changeVisualShape(self.surfaces[0], -1, rgbaColor=table_color, physicsClientId=self.cid)
        if table_color is not None:
            p.changeVisualShape(self.surfaces[1], -1, rgbaColor=table_color, physicsClientId=self.cid)

        p.changeVisualShape(self.objects[0], -1, rgbaColor=blue_color, physicsClientId=self.cid)

    # for initializing pose
    def robot_clear(self, env):
        contacts_gripper = p.getContactPoints(
            env.robot.robot_uid, self.objects[0], physicsClientId=self.cid)
        clear = len(contacts_gripper) == 0
        return clear

    def robot_target_pose(self):
        # desired_pos = self.objs_center_pos[:3] + np.array(self.default_gripper_offset)
        # desired_pos = np.mean([self.p.getBasePositionAndOrientation(obj, physicsClientId=self.cid)[0]
        # for obj in self.objects], axis=0) + np.array(self.params.object_to_gripper)
        desired_pos = self.box_position + np.array(self.params.object_to_gripper)
        # print(self.params.object_to_gripper)
        return (desired_pos, None)

    def robot_target_pose_clear(self):
        raise NotImplementedError

    def _contact(self, uid_A, uid_B):
        return len(p.getContactPoints(bodyA=uid_A, bodyB=uid_B, physicsClientId=self.cid)) > 0

    def _object_in_box(self, env):
        object_in_box = np.any([self._contact(obj, self.surfaces[1]) and
                                self.p.getBasePositionAndOrientation(obj, physicsClientId=self.cid)[0][2] < 0.15 and
                                not self._contact(obj, self.surfaces[0]) for obj in self.objects])
        contact_with_gripper = np.any([self._contact(obj, env.robot.robot_uid) for obj in self.objects])
        return object_in_box and not contact_with_gripper

    def step(self, env, action):
        self.ep_length += 1
        done = False
        reward = 0.

        if self._object_in_box(env):
            self._cnt_obj_in_box += 1
            if self._cnt_obj_in_box >= 5:
                done = True
        else:
            self._cnt_obj_in_box = 0

        if self._contact(self.surfaces[1], env.robot.robot_uid) or \
                np.any([self._contact(self.surfaces[1], obj) and
                        self.p.getBasePositionAndOrientation(obj, physicsClientId=self.cid)[0][2] >
                        0.15 for obj in self.objects]):
            self._contact_penalty += self.penalty_coeff
        if done:
            reward = 1 - np.tanh(self._contact_penalty)
        task_info = {"task_success": False}
        if done:
            task_info["task_success"] = True
        # self._change_object_colors()
        return None, reward, done, task_info

    def eval_step(self, env, action):
        return self.step(env, action)

    def get_state(self):
        '''return np array with all objects pos, orn'''
        block_pos, block_orn = [], []  # TODO numpy empty here
        for uid in self.objects:
            pos, orn = p.getBasePositionAndOrientation(uid, physicsClientId=self.cid)
            block_pos.extend(pos)
            block_orn.extend(orn)
        return np.array([*block_pos, *block_orn])


class PickAndStow2ObjTask(PickAndStowTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.object_files = [opath('acgd_tasks/models/block_blue.urdf'), opath('acgd_tasks/models/block_red.urdf')]
        self.max_num_objects = len(self.object_files)

    def sample_object_pose(self):
        objs_center_pos, orn = self.params.object_pose
        objs_center_pos[2] += self.params.table_pos[2]
        pos_objs, orn_objs = [objs_center_pos], [orn]
        min_obj_dist = self.params.min_obj_dist
        self.num_objects = self.np_random.randint(1, self.max_num_objects + 1)
        for i in range(1, self.num_objects):
            while True:
                self.params.sample_specific("geom/object_pose")
                tmp_pos, tmp_orn = self.params.object_pose
                tmp_pos[2] += self.params.table_pos[2]
                pos_clear = True
                for pos_obj in pos_objs:
                    if np.linalg.norm(pos_obj - tmp_pos) < min_obj_dist:
                        pos_clear = False
                        break
                if pos_clear:
                    pos_objs.append(tmp_pos)
                    orn_objs.append(tmp_orn)
                    break
        for i in range(self.num_objects, self.max_num_objects):
            pos_objs.append(np.array([0, 0, 0]))
            orn_objs.append(orn)
        return pos_objs, orn_objs

    def _change_object_colors(self):
        red_color = self.params.block_red
        blue_color = self.params.block_blue
        table_color = self.params.table_green

        if table_color is not None:
            p.changeVisualShape(self.surfaces[0], -1, rgbaColor=table_color,
                                physicsClientId=self.cid)
        if table_color is not None:
            p.changeVisualShape(self.surfaces[1], -1, rgbaColor=table_color,
                                physicsClientId=self.cid)

        block_colors = [red_color, blue_color]
        self.np_random.shuffle(block_colors)
        p.changeVisualShape(self.objects[0], -1, rgbaColor=block_colors[0], physicsClientId=self.cid)
        p.changeVisualShape(self.objects[1], -1, rgbaColor=block_colors[1], physicsClientId=self.cid)


class BoxSmallShaped(PickAndStowTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cnt_lifting = 0
        self.has_lifted = False

    def reset(self):
        super().reset()
        self._cnt_lifting = 0
        self.has_lifted = False

    def _is_lifted(self, env):
        uid_table = self.surfaces[0]
        if not self._contact(uid_table, self.objects[0]):
            return True
        else:
            return False

    def step(self, env, action):
        done = False
        # self.p.addUserDebugLine([0,0,1], self.box_position + np.array([-0.01,-0.03, 0.07]),
        # lineColorRGB=[1,0,0], lineWidth=2)

        block_pos_blue = self.p.getBasePositionAndOrientation(self.objects[0], physicsClientId=self.cid)[0]
        actual_end_effector_pos = env.robot.get_tcp_pos()

        # gripper_object_distance reward:
        rew_grip_obj = 0.02 * -np.linalg.norm(block_pos_blue - actual_end_effector_pos)

        # object_goal_distance reward:
        rew_obj_goal = 0.02 * -np.linalg.norm((self.box_position + np.array([-0.01, -0.03, 0.07])) - block_pos_blue)
        # print(rew_grip_obj, rew_obj_goal)
        distance_reward = rew_obj_goal + rew_grip_obj

        contact_penalty = 0
        if self._contact(self.surfaces[1], env.robot.robot_uid) or \
                (self._contact(self.surfaces[1], self.objects[0]) and
                 self.p.getBasePositionAndOrientation(self.objects[0], physicsClientId=self.cid)[0][2] > 0.15):
            contact_penalty = -0.1

        lifting_reward = 0
        if self._is_lifted(env):
            self._cnt_lifting += 1
        else:
            self._cnt_lifting = 0
        if self._cnt_lifting >= 3 and not self.has_lifted:
            lifting_reward = 0.5
            self.has_lifted = True

        box_reward = 0
        if self._object_in_box(env):
            self._cnt_obj_in_box += 1
        else:
            self._cnt_obj_in_box = 0
        if self._cnt_obj_in_box >= 5:
            box_reward = 1
        reward = box_reward + distance_reward + lifting_reward + contact_penalty
        done = bool(box_reward)
        task_info = {"task_success": False}
        if done:
            task_info["task_success"] = True
        return None, reward, done, task_info
