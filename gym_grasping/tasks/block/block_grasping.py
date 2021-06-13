"""
BlockTask: pick up a block
"""
from math import pi
import numpy as np
from gym_grasping.tasks.utils import q_dst, pose2tuple, hsv2rgb
from ..utils import opath


class BlockGraspingTask:

    def __init__(self, cid, np_random, p, env_params, table_surface='cork', **kwargs):
        self.cid = cid
        # redefine this in subclasses
        if table_surface == 'cork':
            self.surface_file = "block/models/tables/table_texture.urdf"
        else:
            self.surface_file = "block/models/tables/table_white.urdf"
        self.object_file = "block/models/block_wood.urdf"

        self.object_num = 1
        # take step in case objects touches robot
        self.robot_clear_step = np.array([0, 0, .3])
        # in world coords, assumes robot at origin

        self.robot_workspace_offset = [[-0.25, -0.1, 0.3],
                                       [0.25, 0.21, 0.465]]

        # generic stuff below
        self.np_random = np_random
        self.p = p  # is this needed?
        self.surfaces = []
        self.objects = []
        self.state_vector = []                    # create an vector representation of the state

        self.params = env_params
        self.params.add_variable("table_size", 1, tag="geom")
        self.params.add_variable("table_pos", tag="geom", center=(0, -0.6, 0.117), d=(0, 0, .001))
        self.params.add_variable("table_orn", tag="geom", center=(0, 0, 0, 1), d=(0, 0, 0, 0))
        self.params.add_variable("object_size", tag="geom", center=1, d=0.03)
        self.params.add_variable("object_pose", tag="geom", center=(0, -0.555, 0.017, 0), d=(.2, .07, 0, 2*pi), f=pose2tuple)
        self.params.add_variable("object_to_gripper", tag="geom", center=(0, 0, 0.35), d=(0, 0, 0))
        self.params.add_variable("block_blue", tag="vis", center=(.57, .49, .375), d=(.04, .04, .04), f=hsv2rgb)
        self.params.add_variable("table_green", tag="vis", center=(0.15, .1, .95), d=(.05, .05, .05), f=hsv2rgb)
        self.params.add_variable("block_red", tag="vis", center=(.011, .526, .721), d=(.04, .04, .04), f=hsv2rgb)

    def load_scene(self):
        """Called once per enviroment, and again every 50 runs"""
        self.objects = []

        # static
        table_path = opath(self.surface_file)
        table = self.p.loadURDF(table_path,
                                self.params.table_pos,
                                self.params.table_orn,
                                globalScaling=self.params.table_size,
                                physicsClientId=self.cid)
        self.surfaces = [table]
        # don't load dynamic stuff here because needs resize

    def load_episode(self):
        """Episode specific load, e.g. for different number of blocks"""
        # clear old objects
        for object_id in self.objects:
            self.p.removeBody(object_id, physicsClientId=self.cid)
        self.objects = []
        self.start_pos = []
        # load new objects
        globalScaling = 1
        for i in range(self.object_num):
            if self.object_num == 1:
                pos, orn = self.params.object_pose
                object_file = self.object_file
            else:
                pos = list(self.object_position[i])
                orn = self.p.getQuaternionFromEuler(self.object_orientation[i])
                object_file = self.object_files[i]

            # TODO(max): this is so that we can randomize table hight without
            # execssive falling, this should probably be applied to whole pose
            pos[2] += self.params.table_pos[2]

            size = self.params.object_size * globalScaling
            flags = self.p.URDF_USE_MATERIAL_COLORS_FROM_MTL
            block_uid = self.p.loadURDF(opath(object_file), pos, orn,
                                        globalScaling=size, physicsClientId=self.cid,
                                        flags=flags)
            self.objects.append(block_uid)
            self.start_pos.append((pos, orn))
            # p.changeDynamics(block_uid,-1,restitution=0.5,physicsClientId=self.cid)

    def reset(self):
        """Called once per episode"""
        # episdoe specific variables
        self.ep_length = 0
        self.max_force = 0
        self.num_contacts = 0
        self.start_pos = []
        self.min_distance = 0

        # sample table height offset
        table_pos = self.params.table_pos
        table_orn = self.params.table_orn
        self.p.resetBasePositionAndOrientation(self.surfaces[0], table_pos,
                                               table_orn,
                                               physicsClientId=self.cid)
        # can't do the same for objects here because we want to vary size
        self.load_episode()
        # TODO(max): this _can_ cause segfaults with EGL, write a test
        self._change_object_colors(color_objects=True)

    def reset_from_curriculum(self):
        # TODO(lukas): can this be removed?
        pass

    # TODO(max): rename to cycle colors
    def _change_object_colors(self, color_objects=False):
        """change the block color_factor based on env_params"""
        rgba_s = self.params.table_green
        rgba_b = self.params.block_blue

        if color_objects:
            iter_el = (self.surfaces[0], *self.objects)
            iter_color = (rgba_s, *[rgba_b]*len(self.objects))
        else:
            iter_el = (self.surfaces[0],)
            iter_color = (rgba_s, )

        for uid, rgba in zip(iter_el, iter_color):
            if rgba is not None:
                self.p.changeVisualShape(uid, -1, rgbaColor=rgba, physicsClientId=self.cid)

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

    def robot_target_pose(self):
        pos, _ = self.p.getBasePositionAndOrientation(self.objects[0], physicsClientId=self.cid)
        desired_pos = np.array(pos) + np.array(self.params.object_to_gripper)
        return (desired_pos, None)

    def robot_target_pose_clear(self):
        pos, _ = self.p.getBasePositionAndOrientation(self.objects[0], physicsClientId=self.cid)
        desired_pos = np.array(pos) + self.robot_clear_step
        return (desired_pos, None)

    def step(self, env, action):
        """step the task"""
        self.ep_length += 1
        done = False
        reward = 0.
        info = {}
        #print("ob", self.objects[0])
        for i, blockUid in enumerate(self.objects):
            #print("i", i)
            contacts = self.p.getContactPoints(blockUid, physicsClientId=self.cid)
            self.min_distance = min(
                [self.min_distance, ] + [c[8] for c in contacts])
            self.max_force = max([self.max_force, ] + [c[9] for c in contacts])
            #print("174 block min distance ", self.min_distance)
            # cache link-id if touching robot else -2
            contacts_links_list = [c[4] if c[2] == env.robot.robot_uid else -2 for c in contacts]
            contacts_links = set(contacts_links_list)

            # only touching robot -> managed to lift and contact with two links
            robot_contacts = bool(contacts_links - set((-2,)))
            # robot_contacts_set = set(contacts_links.keys())
            # robot_contacts = bool(robot_contacts_set)
            self.num_contacts += int(robot_contacts)
            # print("contacts", robot_contacts, self.num_contacts)

            if env.robot.contact_callback and robot_contacts:
                first_contact = next(iter(contacts_links - set((-2,))))
                contact = contacts[contacts_links_list.index(first_contact)]
                env.robot.contact_callback(env=env, object_uid=blockUid,
                                           contact=contact, action=action)

            success = (-2 not in contacts_links) and (
                    set(env.robot.gripper.finger_link_ids) <= contacts_links)

            # getBaseVelocitiy

            if not success:
                object_x = self.p.getBasePositionAndOrientation(self.objects[i], physicsClientId=self.cid)
                # print("195 object ", object_x)
                gripper_state = self.p.getLinkState(env.robot.robot_uid,
                        env.robot.gripper_index,
                        computeLinkVelocity=True,
                        physicsClientId=self.cid)
                gripper_x = gripper_state[0:2]
                gripper_v = gripper_state[6:8]
                #print(success)
                #print(type(object_x))
                x = []
                for c in gripper_x[0]:
                    x.append(np.around(c, 2))
                for c in gripper_x[1]:
                    x.append(np.around(c, 2))
            
                v = []
                for c in gripper_v[0]:
                    v.append(np.around(c, 2))
                for c in gripper_v[1]:
                    v.append(np.around(c, 2))
                # print("block gripper position x ", self.state_vector)
                #print(' '.join(format(f, '.3f') for f in gripper_x[0]))
                #print(' '.join(format(f, '.3f') for f in gripper_x[1]))
                o = []
                for c in object_x[0]:
                    o.append(np.around(c, 2))
                for c in object_x[1]:
                    o.append(np.around(c, 2))
                self.state_vector = np.array(x + v + o)
            if success:
                reward = 1.0
                done = True
                # object position before/after
                object_x = self.p.getBasePositionAndOrientation(self.objects[i], physicsClientId=self.cid)
                object_v = self.p.getBaseVelocity(blockUid, physicsClientId=self.cid)
                angle_delta = q_dst(self.start_pos[i][1], object_x[1])
                dist_delta = np.array(self.start_pos[i][0]) - object_x[0]
                dist_delta = np.linalg.norm(dist_delta)

                # gripper velocity <-> object velocity
                gripper_state = self.p.getLinkState(env.robot.robot_uid,
                                                    env.robot.gripper_index,
                                                    computeLinkVelocity=True,
                                                    physicsClientId=self.cid)
                gripper_x = gripper_state[0:2]
                gripper_v = gripper_state[6:8]
                
                x = []
                for c in gripper_x[0]:
                    x.append(np.around(c, 2))
                for c in gripper_x[1]:
                    x.append(np.around(c, 2))
            
                v = []
                for c in gripper_v[0]:
                    v.append(np.around(c, 2))
                for c in gripper_v[1]:
                    v.append(np.around(c, 2))
                
                o = []
                for c in object_x[0]:
                    o.append(np.around(c, 2))
                for c in object_x[1]:
                    o.append(np.around(c, 2))
                
                self.state_vector = np.array(x + v + o)
                #print(' '.join(format(f, '.3f') for f in gripper_x[0]))
                # print("block gripper position", gripper_v)
                diff_v = np.array(gripper_v) - np.array(object_v)
                vel_delta = np.linalg.norm(diff_v, axis=1)
                # normalize coordinate based deltas
                coord_delta = [dist_delta, vel_delta[0], vel_delta[1]]
                coord_delta = np.tanh(coord_delta).sum()
                # normalize combination of coordinate and angle based deltas
                penalty = np.tanh(coord_delta + angle_delta)
                reward = 1.0 - penalty

                info = dict(ep_length=self.ep_length,
                            object_x0=[tuple(x) for x in self.start_pos[i]],
                            object_x=object_x,
                            object_v=object_v,
                            gripper_x=gripper_x,
                            gripper_v=gripper_v,
                            contacts=contacts,
                            max_force=self.max_force,
                            num_contacts=self.num_contacts,
                            min_distance=self.min_distance)
                break
        # step function should return obs, reward, done, info
        # print(self.state_vector)
        return None, reward, done, info

    def get_state(self):
        block_pos, block_orn = self.p.getBasePositionAndOrientation(
            self.objects[0], physicsClientId=self.cid)
        return np.array([*block_pos, *block_orn])


class BlockGraspingTaskShaped(BlockGraspingTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distance_threshold = 0.05

    def step(self, env, action):
        _, _, done, info = super().step(env, action)

        block_pos = self.get_state()[0:3]
        actual_end_effector_pos = env.robot.get_tcp_pos()

        # gripper_object_distance reward:
        reward = -np.linalg.norm(block_pos - actual_end_effector_pos) if not done else 0

        return None, reward, done, info
