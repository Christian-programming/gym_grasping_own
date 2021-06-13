from gym_grasping.tasks.task_names_local import task_names as task_names_local
from gym_grasping.tasks.block.block_grasping import BlockGraspingTask
from gym_grasping.tasks.block.block_grasping import BlockGraspingTaskShaped
from gym_grasping.tasks.block.task_multi import MultiBlockTask
from gym_grasping.tasks.pick_and_place.pick_n_place_task import PickNPlaceTask
from gym_grasping.tasks.bolt.task import BoltTask
from gym_grasping.tasks.acgd_tasks.pick_n_stow_task import (PickAndStowTask,
                                                            PickAndStow2ObjTask)
from gym_grasping.tasks.acgd_tasks.stacking_task import (StackingTask, StackVel,
                                                         StackRewPerStep, StackVelActPen,
                                                         StackRewPerStepAbort,
                                                         StackRewTillEnd, StackShaped)
from gym_grasping.tasks.flow_tasks.flow_tasks import (FlowCalibTask,
                                                      FlowStackingTask,
                                                      FlowStackingTask2)
task_names = {
    'grasp': BlockGraspingTask,
    'multi_block': MultiBlockTask,
    'bolt': BoltTask,
    'grasp_shaped': BlockGraspingTaskShaped,
    'stack': StackingTask,
    'stackVel':  StackVel,
    'stack_shaped':  StackShaped,
    'stackRewPerStep':  StackRewPerStep,
    'stackVelActPen':  StackVelActPen,
    'stackRewPerStepAbort':  StackRewPerStepAbort,
    'stackRewTillEnd':  StackRewTillEnd,
    'pick_n_stow': PickAndStowTask,
    'pick_n_stow_2obj': PickAndStow2ObjTask,
    'flow_calib': FlowCalibTask,
    'flow_stack': FlowStackingTask,
    'flow_stack2': FlowStackingTask2,
    'pick_n_place': PickNPlaceTask,
    **task_names_local
}
