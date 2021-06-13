from gym_grasping.robots.robot_names_local import robot_names as robot_names_local
from gym_grasping.robots.kuka import KukaBolt, KukaSuction

robot_names = {
    'kuka': KukaBolt,
    'suction': KukaSuction,
    **robot_names_local
}
