import numpy as np
import pybullet as p

from .block_grasping import BlockGraspingTask

from ..utils import opath, GLOBAL_SCALE as gS
from gym_grasping.tasks import utils as pbh


class MultiBlockTask(BlockGraspingTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # multiple objectsare supposed nowe
        self.object_file = None  # TODO i would like to have a base class wich
        # supports mulitle object_files
        self.num_objects = 3  # TODO as input arg but changes neeed in grasping all as kwargs
        # call opath here -> more flexible for differnt sources and work only once
        self.object_files = [opath("block/models/block_wood.urdf")] * self.num_objects

    # random intial pose
    def sample_object_pose(self):
        ''' return s  list of pos and orns for each index in self.object_files
            objects are placed randomly  in a grind around the
             object_pos_orn and the params.object_pos_orn_offset
        '''
        base_pos, orn = self.params.object_pose

        pos_objs, orn_objs = [], []
        assert self.object_files is not None, "no files are None"
        assert self.num_objects <= len(self.object_files), \
            "num_objects {} > {}".format(
                self.num_objects, " ".join(self.object_files))

        # DEBUG show grid postions
        pos_xy = self._create_xy_grid(base_pos)
        self._debug_show_poitns(pos_xy * gS, z=base_pos[2] * gS)

        # sample random points, with no replace
        # np_random.choice only for 1d array
        index_options = np.arange(0, len(pos_xy))
        index_rand = self.np_random.choice(
            index_options, self.num_objects, replace=False)
        # append xyz points with globalScaling
        for idx in index_rand:
            xy = pos_xy[idx] * gS
            pos = [*xy, base_pos[2] * gS]
            pos_objs.append(pos)
            orn_objs.append(orn)
        return pos_objs, orn_objs

    def _create_xy_grid(self, base_pos):
        '''returns xy grind points around a base point'''
        # create a X gird around base_pos
        # TODO tune GRID values
        assert len(base_pos) >= 2, "xy point needed"
        # fix range for now
        range_pos_x = 0.03
        range_pos_y = 0.03
        step_dist = 0.01  # block max len + margin (gripper size)
        # TODO check with self.workspace in kuka.py
        # or differnt usage with env_params
        x = np.arange(base_pos[0] - range_pos_x,
                      base_pos[0] + range_pos_x, step_dist)
        y = np.arange(base_pos[1] - range_pos_y,
                      base_pos[1] + range_pos_y, step_dist)
        x, y = np.meshgrid(x, y)
        pos_xy = np.column_stack((x.ravel(), y.ravel()))
        return pos_xy

    def _debug_show_poitns(self, pos_xy, z):
        for xy in pos_xy:
            pbh.add_debug_point([*xy, z], [0, 0, 0, 1],
                                self.p, self.cid, line_len=0.05, life_time=1.)

    def _change_object_colors(self):
        pass  # TODO env_params needs update

    def load_episode(self):
        if len(self.objects):
            for uid in self.objects:
                p.removeBody(uid, physicsClientId=self.cid)
            self.objects = []
        globalScaling = 1
        pos_objs, orn_objs = self.sample_object_pose()

        assert len(pos_objs) == len(self.object_files), \
            "expext a pos and orn for every object_files"
        for mdl_file, pos, orn in zip(self.object_files, pos_objs, orn_objs):
            block_uid = p.loadURDF(mdl_file, pos, orn,
                                   globalScaling=globalScaling, physicsClientId=self.cid)
            self.objects.append(block_uid)
            self.start_pos.append((pos, orn))
            p.changeDynamics(block_uid, -1, restitution=self.params.restitution, physicsClientId=self.cid)

    def get_state(self):
        '''return np array with all objects pos, orn'''
        block_pos, block_orn = [], []  # TODO numpy empty here
        for uid in self.objects:
            pos, orn = p.getBasePositionAndOrientation(uid, physicsClientId=self.cid)
            block_pos.extend(pos)
            block_orn.extend(orn)
        return np.array([*block_pos, *block_orn])
