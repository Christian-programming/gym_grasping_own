from math import pi
import numpy as np
import pybullet as p
from ..utils import opath
from gym_grasping.tasks import utils as pbh
from gym_grasping.tasks.utils import q_dst, pose2tuple, hsv2rgb


class StackingTask:

    def __init__(self, cid, np_random, p, env_params, *args, block_type='primitive', table_surface='cork', **kwargs):
        self.cid = cid
        if table_surface == 'cork':
            self.surface_file = "acgd_tasks/models/tables/table_texture.urdf"
        else:
            self.surface_file = "acgd_tasks/models/tables/table_primitive.urdf"

        # in world coords, assumes robot at origin
        self.robot_workspace_offset = [[-0.25, -0.1, 0.3],
                                       [0.25, 0.21, 0.465]]
        # generic stuff below
        self.np_random = np_random
        self.p = p
        self.surfaces = []
        self.objects = []

        self.params = env_params
        self.params.add_variable("table_size", 1, tag="geom")
        self.params.add_variable("table_pos", tag="geom", center=(0, -0.6, 0.117), d=(0, 0, .001))
        self.params.add_variable("table_orn", tag="geom", center=(0, 0, 0, 1), d=(0, 0, 0, 0))
        self.params.add_variable("object_size", tag="geom", center=1, d=0.03)
        self.params.add_variable("object_pose", tag="geom", center=(0, -0.555, 0.015, 0), d=(.2, .07, 0, 2 * pi), f=pose2tuple)

        # multiple objectsare supposed nowe
        self.object_file = None  # TODO i would like to have a base class wich
        # supports mulitle object_files
        self.num_objects = 3  # TODO as input arg but changes neeed in grasping all as kwargs
        # call opath here -> more flexible for differnt sources and work only once
        self.params.add_variable("min_obj_dist", .03, tag="geom")
        self.params.add_variable("min_steps_full_stack_standing", tag="task",
                                 ll=10, ul=10, f=int)
        self.params.add_variable('max_block_vel', ll=0.001, ul=0.01, tag="task",
                                 mu_s=0.01, mu_e=0.001, r_s=0, r_e=0)
        self.params.add_variable("object_to_gripper", tag="geom",
                                 ll=(-0.04, -0.04, 0.32),
                                 ul=(0.04, 0.04, 0.42),
                                 mu_s=(0, 0, 0.33),
                                 mu_e=(0, 0, 0.38),
                                 r_s=(0.01, 0.01, 0.01),
                                 r_e=(0.04, 0.04, 0.04))
        if table_surface == 'white' or table_surface == 'cork':
            self.params.add_variable("block_blue", tag="vis", center=(.57, .49, .375), d=(.04, .04, .04), f=hsv2rgb)
            self.params.add_variable("table_green", tag="vis", center=(0.15, .1, .95), d=(.05, .05, .05), f=hsv2rgb)
            self.params.add_variable("block_red", tag="vis", center=(.011, .526, .721), d=(.04, .04, .04), f=hsv2rgb)
        else:
            raise ValueError("Invalid or missing argument: table_surface. Passed argument: {}".format(str(table_surface)))

        # multiple objectsare supposed nowe
        # get all mdl files in dir
        if block_type == 'primitive':
            self.object_files = [pbh.get_model_files(opath("acgd_tasks/models"))[i] for i in [0, 2]]
        elif block_type == 'model':
            self.object_files = [pbh.get_model_files(opath("acgd_tasks/models"))[i] for i in [1, 3]]
        elif block_type == 'mixed':
            self.mixed_files = [pbh.get_model_files(opath("acgd_tasks/models"))[i] for i in [0, 2, 1, 3]]
            self.object_files = self.mixed_files[:2]
        else:
            raise Exception
        self.block_type = block_type
        assert len(self.object_files), "no files found"
        self.num_objects = len(self.object_files)
        for i in range(1, self.num_objects):
            self.params.add_variable("block_{}".format(i), tag="geom",
                                     center=(0, 0, 0, 0),
                                     d=(0.065, 0.065, 0, 2*pi),
                                     r_s=(0.05, 0.05, 0, 2*pi),
                                     r_e=(0.065, 0.065, 0.0, 2*pi),
                                     f=pose2tuple)
        # self.num_objects = 2
        assert self.num_objects >= 2, "stack at least 2 objects"
        # counter to check if the full stack stands at least for
        # _min_steps_full_stack_standing steps
        self._cnt_full_stack_standing = 0
        self.params.add_variable("block_type_prob", tag="task", ll=0.0, ul=1.0,
                                 mu_s=0.0, mu_e=1.0, r_s=0.1, r_e=0.1)

    def load_scene(self):
        """Called once per enviroment, and again every 50 runs"""
        self.objects = []

        # static
        table_path = opath(self.surface_file)
        table = self.p.loadURDF(table_path,
                                self.params.table_pos,
                                self.params.table_orn,
                                globalScaling=self.params.table_size, physicsClientId=self.cid)
        self.surfaces = [table]
        # don't load dynamic stuff here because needs resize

    def load_episode(self):
        if len(self.objects):
            for uid in self.objects:
                p.removeBody(uid, physicsClientId=self.cid)
            self.objects = []
        globalScaling = 1
        # TODO(lukas):
        # object_size = self.params.object_size
        pos_objs, orn_objs = self.sample_object_pose()

        assert len(pos_objs) == len(self.object_files), \
            "expext a pos and orn for every object_files"
        for mdl_file, pos, orn in zip(self.object_files, pos_objs, orn_objs):
            block_uid = p.loadURDF(mdl_file, pos, orn,
                                   globalScaling=globalScaling, physicsClientId=self.cid)
            self.objects.append(block_uid)
            self.start_pos.append((pos, orn))
            p.changeDynamics(block_uid, -1, restitution=self.params.restitution, physicsClientId=self.cid)

    def reset_from_curriculum(self):
        self.start_pos = []
        for uid in self.objects:
            self.start_pos.append(p.getBasePositionAndOrientation(uid, physicsClientId=self.cid))

    def reset(self):
        if self.block_type == 'mixed':
            if self.params.block_type_prob < self.np_random.rand():
                self.object_files = self.mixed_files[:2]
            else:
                self.object_files = self.mixed_files[2:]
        """Called once per episode"""
        # episdoe specific variables
        self.ep_length = 0
        self.max_force = 0
        self.num_contacts = 0
        self.start_pos = []
        self.min_distance = 0

        # sample table height offset
        table_pos = self.params.table_pos
        self.p.resetBasePositionAndOrientation(self.surfaces[0], table_pos,
                                               self.p.getQuaternionFromEuler([0, 0, pi / 2]),
                                               physicsClientId=self.cid)
        # can't do the same for objects here because we want to vary size
        self.load_episode()
        # TODO(max): this _can_ cause segfaults with EGL, write a test
        self._change_object_colors()
        self._cnt_full_stack_standing = 0

    def _get_contact_uids(self, uid_A, uids_B):
        '''check for unique contacts of uid_A with a list of uids (uids_B)'''
        contact_uids = []
        for uid_B in uids_B:
            if uid_A != uid_B:
                if len(p.getContactPoints(bodyA=uid_A, bodyB=uid_B, physicsClientId=self.cid)):
                    contact_uids.append(uid_B)

        # print("uid_A ", uid_A, " -> ", contact_uids)
        return contact_uids

    def robot_target_pose(self):
        # desired_pos = self.objs_center_pos[:3] + np.array(self.default_gripper_offset)
        desired_pos = np.mean([self.p.getBasePositionAndOrientation(obj, physicsClientId=self.cid)[0] for obj in self.objects], axis=0) + np.array(self.params.object_to_gripper)
        return (desired_pos, None)

    def _change_object_colors(self):
        red_color = self.params.block_red
        blue_color = self.params.block_blue
        table_color = self.params.table_green

        if table_color is not None:
            p.changeVisualShape(self.surfaces[0], -1, rgbaColor=table_color, physicsClientId=self.cid)
        if red_color is not None:
            p.changeVisualShape(self.objects[0], -1, rgbaColor=red_color, physicsClientId=self.cid)
        if blue_color is not None:
            p.changeVisualShape(self.objects[1], -1, rgbaColor=blue_color, physicsClientId=self.cid)

    # random intial pose
    def sample_object_pose(self):
        ''' returns  list of pos and orns for each index in self.object_files
            objects are placed randomly in a grid around the
            object_pos_orn and the params.object_pos_orn_offset
        '''
        objs_center_pos, orn = self.params.object_pose
        objs_center_pos[2] += self.params.table_pos[2]

        pos_objs, orn_objs = [objs_center_pos], [orn]
        assert self.object_files is not None, "no files are None"
        assert self.num_objects <= len(self.object_files), \
            "num_objects {} > {}".format(
                self.num_objects, " ".join(self.object_files))
        min_obj_dist = self.params.min_obj_dist

        for i in range(1, len(self.object_files)):
            while True:
                self.params.sample_specific("block_{}".format(i))
                tmp_offset, tmp_orn = self.params.sample["block_{}".format(i)]
                # tmp_offset, tmp_orn = self.params.sample["block_{}".format(i)]
                tmp_xy = objs_center_pos + tmp_offset
                pos_clear = True
                for pos_obj in pos_objs:
                    if np.linalg.norm(pos_obj - tmp_xy) < min_obj_dist:
                        pos_clear = False
                        break
                if pos_clear:
                    pos_objs.append(tmp_xy)
                    orn_objs.append(tmp_orn)
                    break
        return pos_objs, orn_objs

    def _is_stacked(self, env):
        '''
        If all objects stacked on each other the reward is 1, else 0.
        reward based on contacts:
            table <-> block0  <-> block1 <-> block2 =>  success
        '''
        uid_table = self.surfaces[0]
        contact_table = self._get_contact_uids(uid_table, self.objects)
        # check if one block touches the table
        if len(contact_table) == 1:
            # objects are stack all on top of each other if,
            # we have the contact two with 1 contact and all others have two
            contact_objects = [self._get_contact_uids(
                uid, self.objects) for uid in self.objects]
            num_one_contacts = sum(1 for uids in contact_objects if len(uids) == 1)
            num_two_contacts = sum(1 for uids in contact_objects if len(uids) == 2)
            has_contact_with_gripper = len(self._get_contact_uids(env.robot.robot_uid, self.objects)) > 0
            if num_one_contacts == 2 and num_two_contacts == self.num_objects - 2 and not has_contact_with_gripper:
                return True
        else:
            return False

    def _calculate_movement_penalty(self, block_uid):
        object_x = p.getBasePositionAndOrientation(block_uid, physicsClientId=self.cid)
        # object_v = p.getBaseVelocity(block_uid)

        i = self.objects.index(block_uid)

        angle_delta = q_dst(self.start_pos[i][1], object_x[1])
        dist_delta = np.linalg.norm(self.start_pos[i][0]
                                    - np.array(object_x[0]))
        return np.tanh(dist_delta*10 + angle_delta)

    def set_uid_color(self, uid, rgba, link_id=-1):
        """
        helper function used by notebooks.
        """
        self.p.changeVisualShape(uid, link_id, rgbaColor=rgba, physicsClientId=self.cid)

    # for initializing pose
    def robot_clear(self, env):
        """
        Test if a robot is clear of contacts

        Returns:
            clear: bool True is clear, False not clear
        """
        contacts_gripper = self.p.getContactPoints(
            env.robot.robot_uid, self.objects[0], physicsClientId=self.cid)
        clear = len(contacts_gripper) == 0
        return clear

    def _objects_moving(self):
        object_vs = []
        for uid in self.objects:
            object_vs.append(np.mean(p.getBaseVelocity(uid, physicsClientId=self.cid)[0]))
        return not (np.abs(object_vs) < self.params.max_block_vel).all()

    def eval_step(self, env, action):
        reward = 0
        if self._is_stacked(env):
            self._cnt_full_stack_standing += 1
        else:
            self._cnt_full_stack_standing = 0
        if self._cnt_full_stack_standing >= self.params.min_steps_full_stack_standing and not self._objects_moving():
            reward = 1 - self._calculate_movement_penalty(self.objects[0])
        done = bool(reward)
        task_info = {"task_success": False}
        if done:
            task_info["task_success"] = True
        return None, reward, done, task_info

    def step(self, env, action):
        reward = 0
        if self._is_stacked(env):
            self._cnt_full_stack_standing += 1
        else:
            self._cnt_full_stack_standing = 0
        if self._cnt_full_stack_standing >= self.params.min_steps_full_stack_standing and not self._objects_moving():
            reward = 1
        done = bool(reward)
        task_info = {"task_success": False}
        if done:
            task_info["task_success"] = True
        return None, reward, done, task_info

    def get_state(self):
        block_pos, block_orn = self.p.getBasePositionAndOrientation(
            self.objects[0], physicsClientId=self.cid)
        return np.array([*block_pos, *block_orn])


class StackVel(StackingTask):
    def step(self, env, action):
        return super().eval_step(env, action)


class StackRewPerStep(StackingTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_gripper_action = 1

    def reset(self):
        super().reset()
        self.prev_gripper_action = 1

    def _calculate_action_penalty(self, env, action):
        gripper_action = 1 if action[4] > env.robot.gripper.closing_threshold else -1
        movement_penalty = np.linalg.norm(action[:4]) * 0.0001
        gripper_penalty = 0.002 if gripper_action != self.prev_gripper_action else 0
        self.prev_gripper_action = gripper_action
        action_penalty = -(movement_penalty + gripper_penalty)
        return action_penalty

    def _calculate_block_movement_penalty(self, block_uid):
        object_v = p.getBaseVelocity(block_uid, physicsClientId=self.cid)
        cartesian_v = np.linalg.norm(object_v[0])
        angular_v = 0.02 * np.linalg.norm(object_v[1])
        return -(angular_v + cartesian_v)

    def step(self, env, action):
        # action penalty
        action_penalty = self._calculate_action_penalty(env, action)
        k = self.objects[0]
        #print("ob", k)
        # block movement penalty
        block_movement_penalty = 0
        if env._ep_step_counter > 1:
            block_movement_penalty = 0.02 * self._calculate_block_movement_penalty(self.objects[0])

        stacking_reward = 0
        if self._is_stacked(env):
            self._cnt_full_stack_standing += 1
        else:
            self._cnt_full_stack_standing = 0
        if self._cnt_full_stack_standing >= self.params.min_steps_full_stack_standing and not self._objects_moving():
            stacking_reward = 1
        reward = stacking_reward + action_penalty + block_movement_penalty

        done = bool(stacking_reward)
        task_info = {"task_success": False}
        if done:
            task_info["task_success"] = True
        return None, reward, done, task_info


class StackVelActPen(StackRewPerStep):
    def step(self, env, action):
        # action penalty
        action_penalty = self._calculate_action_penalty(env, action)

        stacking_reward = 0
        if self._is_stacked(env):
            self._cnt_full_stack_standing += 1
        else:
            self._cnt_full_stack_standing = 0
        if self._cnt_full_stack_standing >= self.params.min_steps_full_stack_standing and not self._objects_moving():
            stacking_reward = 1 - self._calculate_movement_penalty(self.objects[0])
        reward = stacking_reward + action_penalty

        done = bool(stacking_reward)
        task_info = {"task_success": False}
        if done:
            task_info["task_success"] = True
        return None, reward, done, task_info


class StackRewPerStepAbort(StackRewPerStep):
    def step(self, env, action):
        # action penalty
        action_penalty = self._calculate_action_penalty(env, action)

        # block movement penalty
        block_movement_penalty = 0
        if env._ep_step_counter > 1:
            block_movement_penalty = 0.02 * self._calculate_block_movement_penalty(self.objects[0])

        done = False
        success = False
        stacking_reward = 0
        if self._is_stacked(env):
            self._cnt_full_stack_standing += 1
            stacking_reward = np.clip(0.1 + 0.4 * self._calculate_block_movement_penalty(self.objects[1]), 0, 0.1)
            if self._cnt_full_stack_standing >= self.params.min_steps_full_stack_standing():
                done = True
                if not self._objects_moving():
                    success = True
        elif self._cnt_full_stack_standing > 0:
            done = True

        reward = stacking_reward + action_penalty + block_movement_penalty
        task_info = dict(task_success=success)
        return None, reward, done, task_info


class StackRewTillEnd(StackRewPerStep):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.success = False

    def reset(self):
        super().reset()
        self.success = False

    def step(self, env, action):
        # action penalty
        action_penalty = self._calculate_action_penalty(env, action)

        # block movement penalty
        block_movement_penalty = 0
        if env._ep_step_counter > 1:
            block_movement_penalty = 0.02 * self._calculate_block_movement_penalty(self.objects[0])
        done = False
        stacking_reward = 0
        if self._is_stacked(env):
            self._cnt_full_stack_standing += 1
            stacking_reward = np.clip(0.1 + 0.4 * self._calculate_block_movement_penalty(self.objects[1]), 0, 0.1)
            if self._cnt_full_stack_standing and not self._objects_moving():
                self.success = True
        elif self._cnt_full_stack_standing > 0:
            self._cnt_full_stack_standing = 0
        reward = stacking_reward + action_penalty + block_movement_penalty

        task_info = {"task_success": self.success}
        return None, reward, done, task_info


class StackShaped(StackRewPerStep):
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
        contact_table = self._get_contact_uids(uid_table, self.objects)
        # check if one block touches the table
        if len(contact_table) == 1 and 2 in contact_table:
            return True
        else:
            return False

    def step(self, env, action):
        # action penalty
        action_penalty = self._calculate_action_penalty(env, action)

        done = False

        block_pos_red = self.get_state()[0:3]
        block_pos_blue = self.get_state()[3:6]
        actual_end_effector_pos = env.robot.get_tcp_pos()

        # gripper_object_distance reward:
        rew_grip_obj = 0.02 * -np.linalg.norm(block_pos_blue - actual_end_effector_pos)

        # object_goal_distance reward:
        rew_obj_goal = 0.02 * -np.linalg.norm((block_pos_red + np.array([0, 0, 0.06])) - block_pos_blue)
        # print(rew_grip_obj, rew_obj_goal)
        distance_reward = rew_obj_goal + rew_grip_obj
        # print("blocked_pos")
        lifting_reward = 0
        if self._is_lifted(env):
            self._cnt_lifting += 1
        else:
            self._cnt_lifting = 0
        if self._cnt_lifting >= 3 and not self.has_lifted:
            lifting_reward = 0.5
            self.has_lifted = True

        stacking_reward = 0
        if self._is_stacked(env):
            self._cnt_full_stack_standing += 1
        else:
            self._cnt_full_stack_standing = 0
        if self._cnt_full_stack_standing >= self.params.min_steps_full_stack_standing and not self._objects_moving():
            stacking_reward = 1
        reward = stacking_reward + action_penalty + distance_reward + lifting_reward
        done = bool(stacking_reward)
        task_info = {"task_success": False}
        if done:
            task_info["task_success"] = True
        return None, reward, done, task_info
