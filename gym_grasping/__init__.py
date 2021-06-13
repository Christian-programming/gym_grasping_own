from gym.envs.registration import register

register(
    id='KukaBinpickState-v0',
    entry_point='gym_grasping.envs.robot_sim_env:RobotSimEnv',
    kwargs={'act_type': 'multi-discrete', 'obs_type': 'state'}
)

register(
    id='KukaBinpickDiscreteEasy-v0',
    entry_point='gym_grasping.envs.robot_sim_env:RobotSimEnv',
    kwargs={'act_type': 'discrete', 'initial_pose': 'close'}
)

register(
    id='KukaBinpickDiscrete-v0',
    entry_point='gym_grasping.envs.robot_sim_env:RobotSimEnv',
    kwargs={'act_type': 'discrete'}
)

# Bolt Envs (continous)
register(
    id='KukaBolt-v0',
    entry_point='gym_grasping.envs.robot_sim_env:RobotSimEnv',
    kwargs={'act_type': 'continuous', 'initial_pose': 'close', 'task': 'bolt'}
)

# Bolt Envs (multi-discrete)
register(
    id='KukaBoltMD-v0',
    entry_point='gym_grasping.envs.robot_sim_env:RobotSimEnv',
    kwargs={'act_type': 'multi-discrete', 'initial_pose': 'close', 'task': 'bolt'}
)


# Bolt Envs (discrete)
register(
    id='KukaBoltDiscreteEasy-v0',
    entry_point='gym_grasping.envs.robot_sim_env:RobotSimEnv',
    kwargs={'act_type': 'discrete', 'initial_pose': 'close', 'task': 'bolt'}
)


# Block Envs (multi-discrete)
register(
    id='KukaBlock-v0',
    entry_point='gym_grasping.envs.robot_sim_env:RobotSimEnv',
    kwargs={'act_type': 'continuous', 'initial_pose': 'close', 'task': 'grasp'}
)

register(
    id='KukaBlockMD-v0',
    entry_point='gym_grasping.envs.robot_sim_env:RobotSimEnv',
    kwargs={'act_type': 'multi-discrete', 'initial_pose': 'close', 'task': 'grasp'}
)


#
# registered tasks using EGL
register(
    id='kuka_block_cont_img-v0',
    entry_point='gym_grasping.envs.robot_sim_env:RobotSimEnv',
    kwargs={'act_type': 'continuous', 'initial_pose': 'close', 'robot': 'kuka', 'task': 'grasp',
            'obs_type': 'image', 'renderer': 'egl', 'act_dv': 0.001,
            'img_type': 'rgb'}
)

register(
    id='kuka_block_cont_combi-v0',
    entry_point='gym_grasping.envs.robot_sim_env:RobotSimEnv',
    kwargs={'act_type': 'continuous', 'initial_pose': 'close', 'robot': 'kuka', 'task': 'grasp', 'obs_type': 'image_state',
            'renderer': 'egl', 'act_dv': 0.001,
            'img_type': 'rgb'}
)

# block grasping
register(
    id='kuka_block_grasping-v0',
    entry_point='gym_grasping.envs.robot_sim_env:RobotSimEnv',
    kwargs={'act_type': 'continuous', 'initial_pose': 'close', 'task': 'grasp_shaped',
            'obs_type': 'image_state', 'renderer': 'egl', 'max_steps': 50, 'camera_pos': 'new_mount'}
)

#
# stacking
register(
    id='kuka_block_stacking-v0',
    entry_point='gym_grasping.envs.robot_sim_env:RobotSimEnv',
    kwargs={'act_type': 'continuous', 'initial_pose': 'close', 'task': 'stack_shaped',
            'obs_type': 'image_state', 'renderer': 'egl', 'max_steps': 150, 'camera_pos': 'new_mount'}
)


register(
    id='stackVel_acgd-v0',
    entry_point='gym_grasping.envs.curriculum_env:AdaptiveCurriculumEnv',
    kwargs={'curr': 'stack', 'initial_pose': 'close', 'task': 'stackVel', 'max_steps': 150,
            'use_regular_starts': True, 'reg_start_func': 'f_sr_prog', 'renderer': 'egl',
            'camera_pos': 'new_mount', 'obs_type': 'image_state_reduced', 'adaptive_task_difficulty': True,
            "diff": "diff_cur_reg"}
)

# stacking multi-discrete
register(
    id='stackVel_acgd_md_fast-v0',
    entry_point='gym_grasping.envs.curriculum_env:AdaptiveCurriculumEnv',
    kwargs={'curr': 'stack', 'initial_pose': 'close', 'task': 'stackVel', 'max_steps': 150,
            'use_regular_starts': True, 'reg_start_func': 'f_sr_prog', 'renderer': 'egl',
            'camera_pos': 'new_mount', 'obs_type': 'image_state_reduced', 'adaptive_task_difficulty': True,
            "diff": "diff_cur_reg", 'act_type': 'multi-discrete'}
)

register(
    id='stackVel_acgd_md_slow1-v0',
    entry_point='gym_grasping.envs.curriculum_env:AdaptiveCurriculumEnv',
    kwargs={'curr': 'stack', 'initial_pose': 'close', 'task': 'stackVel', 'max_steps': 150,
            'use_regular_starts': True, 'reg_start_func': 'f_sr_prog', 'renderer': 'egl',
            'camera_pos': 'new_mount', 'obs_type': 'image_state_reduced', 'adaptive_task_difficulty': True,
            "diff": "diff_cur_reg", 'param_info': {'robot_dv': {'center': .0005, 'd': .0001},
                                                   "robot_drot": {'center': 0.01, 'd': 0}},
            'act_type': 'multi-discrete'}
)

register(
    id='stackVel_acgd_md_slow2-v0',
    entry_point='gym_grasping.envs.curriculum_env:AdaptiveCurriculumEnv',
    kwargs={'curr': 'stack', 'initial_pose': 'close', 'task': 'stackVel', 'max_steps': 150,
            'use_regular_starts': True, 'reg_start_func': 'f_sr_prog', 'renderer': 'egl',
            'camera_pos': 'new_mount', 'obs_type': 'image_state_reduced', 'adaptive_task_difficulty': True,
            "diff": "diff_cur_reg", 'param_info': {'robot_dv': {'center': .0005, 'd': .0001},
                                                   "robot_drot": {'center': 0.005, 'd': 0}},
            'act_type': 'multi-discrete'}
)
# pick_n_stow
register(
    id='box_acgd-v0',
    entry_point='gym_grasping.envs.curriculum_env:AdaptiveCurriculumEnv',
    kwargs={'curr': 'box', 'initial_pose': 'close', 'task': 'pick_n_stow', 'max_steps': 150,
            'use_regular_starts': True, 'reg_start_func': 'f_sr_prog', 'renderer': 'egl',
            'camera_pos': 'new_mount', 'obs_type': 'image_state_reduced', 'adaptive_task_difficulty': True,
            "diff": "diff_cur_reg"}
)
