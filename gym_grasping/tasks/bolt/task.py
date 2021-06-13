from math import pi
from ..block.block_grasping import BlockGraspingTask
from gym_grasping.tasks.utils import pose2tuple


class BoltTask(BlockGraspingTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.object_file = "bolt/models/bolt.urdf"

        self.params.add_variable("object_pose", tag="geom",
                                 center=(0, -0.55, 0.15, 100/100*pi),
                                 d=(.2, .07, 0, 0), f=pose2tuple)
        self.params.add_variable("object_to_gripper", tag="geom",
                                 center=(0, 0, 0.32), d=(0, 0, 0))

# This should contain a box with many screws.
# class MultiBoltTask(BlockTask):
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)
#        self.object_file = "bolt/models/bolt.urdf"
#        # self.surface_file = "block/models/tray/traybox.urdf"
#        self.surface_size = .3
#        self.object_num = 5
#
#        # self.object_pos_orn = np.array([-.01, -0.71, 0.15, 0])
#        self.params.add_variable("object_pose", tag="geom",
#                                    center=(-.q05, -0.55, 0.15, 100/100*pi),
#                                    d=(.1, .1, 0, 3), f=pose2tuple)
#        self.params.add_variable("object_to_gripper", tag="geom",
#                                    center=(0, 0, 0.32), d=(0, 0, 0))
