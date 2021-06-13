from math import pi
from ..utils import opath
from gym_grasping.tasks.utils import pose2tuple
from ..acgd_tasks.stacking_task import StackingTask
from ..block.block_grasping import BlockGraspingTask
from gym_grasping.calibration.random_pose_sampler import RandomPoseSampler


class FlowCalibTask(BlockGraspingTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.object_file = "calibration/models/tag.urdf"

        self.params.add_variable("robot_dv", center=.001, d=0, tag="dyn")
        self.params.add_variable("gripper_speed", center=10, d=00, tag="dyn")
        self.params.add_variable('gripper_joint_ul', center=0.0455, d=0, tag="dyn")
        self.params.add_variable("gripper_rot", center=0, d=0, tag="dyn")
        # self.params.add_variable("object_to_gripper", tag="geom",
        #                            center=(0, 0, 0.38), d=(0, 0, 0))
        self.pose_sampler = RandomPoseSampler()

    def _change_object_colors(self):
        pass

    def policy(self, env, action=None):
        pose = self.pose_sampler.sample_pose()
        return pose+(1, ), False


class FlowStackingTask(StackingTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.surface_file = "flow_tasks/models/table_textured/table_wood.urdf"
        self.params.add_variable("object_pose", tag="geom",
                                 center=(0.6, -0.555, 0.015, 0),
                                 d=(0, 0, 0, 0),
                                 f=pose2tuple)
        self.params.add_variable("object_to_gripper", tag="geom",
                                 center=(0, 0, 0.38), d=(0, 0, 0))
        self.params.add_variable("table_pos", tag="geom",
                                 center=(0.6, -0.6, 0.117), d=(0, 0, 0))
        self.params.add_variable("block_1", center=(0.05, 0, 0, 0), tag="geom",
                                 d=(0, 0, 0, 0),
                                 r_s=(0.05, 0.05, 0, 2 * pi),
                                 r_e=(0.065, 0.065, 0.0, 2 * pi),
                                 f=pose2tuple)
        self.params.add_variable("robot_base", tag="geom",
                                 center=(-.1 + 0.6, 0, 0.07), d=(0, 0, 0.0))
        self.params.add_variable("robot_dv", center=.001, d=0, tag="dyn")
        self.params.add_variable("gripper_speed", center=10, d=00, tag="dyn")
        self.params.add_variable('gripper_joint_ul', center=0.0455, d=0, tag="dyn")
        self.params.add_variable("gripper_rot", center=0, d=0, tag="dyn")
        self.object_files = [
            opath("flow_tasks/models/block_textured/block_textured1.urdf"),
            opath("flow_tasks/models/block_textured/block_textured2.urdf")]

    def _change_object_colors(self):
        pass


class FlowStackingTask2(StackingTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.surface_file = "flow_tasks/models/lena/table_wood.urdf"
        self.params.add_variable("object_pose", tag="geom",
                                 center=(0.6, -0.555, 0.015, 0),
                                 d=(0, 0, 0, 0),
                                 f=pose2tuple)
        self.params.add_variable("object_to_gripper", tag="geom",
                                 center=(0, 0, 0.38), d=(0, 0, 0))

        self.params.add_variable("table_pos", tag="geom",
                                 center=(0.6, -0.6, 0.117), d=(0, 0, 0))
        self.params.add_variable("block_1", center=(0.05, 0, 0, 0), tag="geom",
                                 d=(0, 0, 0, 0),
                                 r_s=(0.05, 0.05, 0, 2 * pi),
                                 r_e=(0.065, 0.065, 0.0, 2 * pi),
                                 f=pose2tuple)
        self.params.add_variable("robot_base", tag="geom",
                                 center=(-.1 + 0.6, 0, 0.07), d=(0, 0, 0.0))
        self.params.add_variable("robot_dv", center=.001, d=0, tag="dyn")
        self.params.add_variable("gripper_speed", center=10, d=00, tag="dyn")
        self.params.add_variable('gripper_joint_ul', center=0.0455, d=0, tag="dyn")
        self.params.add_variable("gripper_rot", center=0, d=0, tag="dyn")
        self.object_files = [
            opath("flow_tasks/models/block_textured/block_textured1.urdf"),
            opath("flow_tasks/models/block_textured/block_textured2.urdf")]

    def _change_object_colors(self):
        pass
