"""
This is a placeholder, put your local task names here
"""

from gym_grasping.tasks.sick.task import VacuumTask, WindowTask
from gym_grasping.tasks.sick.task_cad_demo import SICKCADTask


task_names = {
    'CAD': SICKCADTask,
    'vacuum': VacuumTask,
    'window': WindowTask
}
