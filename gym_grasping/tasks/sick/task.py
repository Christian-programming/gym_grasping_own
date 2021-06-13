from math import pi
from gym_grasping.tasks.block.block_grasping import BlockGraspingTask
from gym_grasping.tasks.pick_and_place.pick_n_place_task import PickNPlaceTask

from gym_grasping.tasks.utils import pose2tuple, hsv2rgb


class VacuumTask(PickNPlaceTask):
    def __init__(self, cid, np_random, p, env_params, **kwargs):
        super().__init__(cid, np_random, p, env_params, **kwargs)

        self.surface_file = "sick/models/scene_vacuum.urdf"
        self.object_num = 1

        orn = p.getQuaternionFromEuler([-.50, 0, .05])
        self.params.add_variable("table_orn", tag="geom", center=orn, d=(0, 0, 0, 0))
        self.params.add_variable("table_pos", tag="geom", center=(.3, -0.4, .5), d=(0, 0, .001))
        self.params.add_variable("table_green", tag="vis", center=(0.1, .1, 1.0), d=(.05, .05, .05), f=hsv2rgb)

        self.params.add_variable("object_pose", tag="geom", center=(.3, .4, -.4, 0), d=(.2, .07, 0, pi/4), f=pose2tuple)

        self.object_file = "/home/argusm/models/CAD/SICK/WL24-color.urdf"

        # relative to table
        self.robot_workspace_offset = [[-.9, .6, -.2],
                                       [0.3, 1.0, 0.2]]


class WindowTask(BlockGraspingTask):
    def __init__(self, cid, np_random, p, env_params, **kwargs):
        super().__init__(cid, np_random, p, env_params, **kwargs)

        self.surface_file = "sick/models/scene_window.urdf"

        orn = p.getQuaternionFromEuler([-.25, 0, 0])
        self.params.add_variable("table_orn", tag="geom", center=orn, d=(0, 0, 0, 0))
        self.params.add_variable("table_pos", tag="geom", center=(0, -0.7, .5), d=(0, 0, .001))
        self.params.add_variable("table_green", tag="vis", center=(0.1, .1, 1.0), d=(.05, .05, .05), f=hsv2rgb)
