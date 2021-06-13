"""
Define Curricula
these say how the difficulty of individual parameters should be scaled by with
a given global difficulty.
"""


class CurriculumStub:
    '''Base method for indentification'''


class Easy(CurriculumStub):
    """
    No domain randomization
    """

    def set_difficulty(self, env_params, diff, do_regular_start=None):
        for k in env_params.variables:
            env_params.set_variable_difficulty_r(k, 0)
            env_params.set_variable_difficulty_mu(k, 0)


class ChangingParamsBox(CurriculumStub):
    """increase difficulty for pick-n-stow task"""

    @staticmethod
    def set_difficulty(env_params, diff, do_regular_restart=None):
        '''set the difficulty'''
        # appearance
        env_params.set_variable_difficulty_r("vis/block_red", diff)
        env_params.set_variable_difficulty_r("vis/block_blue", diff)
        env_params.set_variable_difficulty_r("vis/table_green", diff)
        env_params.set_variable_difficulty_r("vis/brightness", diff)
        env_params.set_variable_difficulty_r("vis/contrast", diff)
        env_params.set_variable_difficulty_r("vis/color", diff)
        env_params.set_variable_difficulty_r("vis/shaprness", diff)
        env_params.set_variable_difficulty_r("vis/blur", diff)
        env_params.set_variable_difficulty_r("vis/hue", diff)
        env_params.set_variable_difficulty_r("vis/light_direction", diff)

        # cam
        env_params.set_variable_difficulty_r("cam/calib_pos", diff)
        env_params.set_variable_difficulty_r("cam/calib_orn", diff)
        env_params.set_variable_difficulty_r("cam/cam_fov", diff)

        # task difficulty
        env_params.set_variable_difficulty_mu("sim/restitution", diff)
        env_params.set_variable_difficulty_r("geom/object_to_gripper", diff)
        env_params.set_variable_difficulty_mu("geom/object_to_gripper", diff)
        env_params.set_variable_difficulty_r("geom/object_pose", diff)
        env_params.set_variable_difficulty_mu("geom/object_pose", diff)

        # dynamics and scene randomization
        env_params.set_variable_difficulty_r("geom/table_pos", diff)
        env_params.set_variable_difficulty_r("geom/robot_base", diff)
        env_params.set_variable_difficulty_r("gripper_rot", diff)
        env_params.set_variable_difficulty_r("geom/object_size", diff)
        env_params.set_variable_difficulty_r("dyn/robot_dv", diff)
        env_params.set_variable_difficulty_r("gripper_speed", diff)
        env_params.set_variable_difficulty_r("gripper_joint_ul", diff)


class ChangingParamsStack(CurriculumStub):
    """increase difficulty for stacking task"""

    @staticmethod
    def set_difficulty(env_params, diff, do_regular_restart=None):
        '''set the difficulty'''
        # appearance
        env_params.set_variable_difficulty_r("vis/block_red", diff)
        env_params.set_variable_difficulty_r("vis/block_blue", diff)
        env_params.set_variable_difficulty_r("vis/table_green", diff)
        env_params.set_variable_difficulty_r("vis/brightness", diff)
        env_params.set_variable_difficulty_r("vis/contrast", diff)
        env_params.set_variable_difficulty_r("vis/color", diff)
        env_params.set_variable_difficulty_r("vis/shaprness", diff)
        env_params.set_variable_difficulty_r("vis/blur", diff)
        env_params.set_variable_difficulty_r("vis/hue", diff)
        env_params.set_variable_difficulty_r("vis/light_direction", diff)

        # cam
        env_params.set_variable_difficulty_r("cam/calib_pos", diff)
        env_params.set_variable_difficulty_r("cam/calib_orn", diff)
        env_params.set_variable_difficulty_r("cam/cam_fov", diff)

        # task difficulty
        env_params.set_variable_difficulty_mu("sim/restitution", diff)
        env_params.set_variable_difficulty_mu("task/max_block_vel", diff)
        env_params.set_variable_difficulty_mu("task/block_type_prob", diff)
        env_params.set_variable_difficulty_r("geom/object_to_gripper", diff)
        env_params.set_variable_difficulty_mu("geom/object_to_gripper", diff)
        env_params.set_variable_difficulty_r("geom/block_1", diff)

        # dynamics and scene randomization
        env_params.set_variable_difficulty_r("geom/table_pos", diff)
        env_params.set_variable_difficulty_r("geom/robot_base", diff)
        env_params.set_variable_difficulty_r("gripper_rot", diff)
        env_params.set_variable_difficulty_r("geom/object_size", diff)
        env_params.set_variable_difficulty_r("dyn/robot_dv", diff)
        env_params.set_variable_difficulty_r("gripper_speed", diff)
        env_params.set_variable_difficulty_r("gripper_joint_ul", diff)


class SegCurr(CurriculumStub):
    """increase difficulty for training with segmentation"""

    @staticmethod
    def set_difficulty(env_params, diff, do_regular_restart=None):
        '''set the difficulty'''
        # cam
        env_params.set_variable_difficulty_r("cam/calib_pos", diff)
        env_params.set_variable_difficulty_r("cam/calib_orn", diff)
        env_params.set_variable_difficulty_r("cam/cam_fov", diff)

        # task difficulty
        env_params.set_variable_difficulty_mu("sim/restitution", diff)
        env_params.set_variable_difficulty_mu("task/max_block_vel", diff)
        env_params.set_variable_difficulty_mu("task/block_type_prob", diff)
        env_params.set_variable_difficulty_r("geom/object_to_gripper", diff)
        env_params.set_variable_difficulty_mu("geom/object_to_gripper", diff)
        env_params.set_variable_difficulty_r("geom/block_1", diff)

        # dynamics and scene randomization
        env_params.set_variable_difficulty_r("geom/table_pos", diff)
        env_params.set_variable_difficulty_r("geom/robot_base", diff)
        env_params.set_variable_difficulty_r("gripper_rot", diff)
        env_params.set_variable_difficulty_r("geom/object_size", diff)
        env_params.set_variable_difficulty_r("dyn/robot_dv", diff)
        env_params.set_variable_difficulty_r("gripper_speed", diff)
        env_params.set_variable_difficulty_r("gripper_joint_ul", diff)


CURRICULA = {
    'easy': Easy,
    'box': ChangingParamsBox,
    'stack': ChangingParamsStack,
    'seg_cur': SegCurr
}


#
# Restart Functions that determine how to mix regular and demonstration resets
def f_eval_prog(diff, prog, eval_rew, curr_rew, reg_rew, eval_reg_rew, curr_sr, reg_sr):
    '''mix of eval reward and progress'''
    return 0.5 * eval_rew + 0.5 * prog


def f_reg_prog(diff, prog, eval_rew, curr_rew, reg_rew, eval_reg_rew, curr_sr, reg_sr):
    '''mix of reg reward and progress'''
    return 0.5 * reg_rew + 0.5 * prog


def f_eval_reg_prog(diff, prog, eval_rew, curr_rew, reg_rew, eval_reg_rew, curr_sr, reg_sr):
    '''mix of eval regular reward and progress'''
    return 0.5 * eval_reg_rew + 0.5 * prog


def f_eval_reg_max_prog(diff, prog, eval_rew, curr_rew, reg_rew, eval_reg_rew, curr_sr, reg_sr):
    '''mix of max of regular reward and eval reward, and progress'''
    return 0.5 * max(reg_rew, eval_rew) + 0.5 * prog


def f_sr_prog(diff, prog, eval_rew, curr_rew, reg_rew, eval_reg_rew, curr_sr, reg_sr):
    '''mix of success rate and progress'''
    return 0.5 * reg_sr + 0.5 * prog


def f_sr_prog_max(diff, prog, eval_rew, curr_rew, reg_rew, eval_reg_rew, curr_sr, reg_sr):
    '''max of regular success rate and progress'''
    return max(reg_sr, prog)


def f_diff_cur(diff, prog, eval_rew, curr_rew, reg_rew, eval_reg_rew, curr_sr, reg_sr):
    '''difficulty'''
    return diff


def f_diff_prog(diff, prog, eval_rew, curr_rew, reg_rew, eval_reg_rew, curr_sr, reg_sr):
    '''difficulty and progress'''
    return 0.5 * (diff + prog)


def f_only_reg(diff, prog, eval_rew, curr_rew, reg_rew, eval_reg_rew, curr_sr, reg_sr):
    '''only regular resarts'''
    return 1


RESTART_FUNCS = {
    'f_eval_prog': f_eval_prog,
    'f_reg_prog': f_reg_prog,
    'f_eval_reg_prog': f_eval_reg_prog,
    'f_eval_reg_max_prog': f_eval_reg_max_prog,
    'f_sr_prog': f_sr_prog,
    'f_sr_prog_max': f_sr_prog_max,
    'f_diff_cur': f_diff_cur,
    'f_diff_prog': f_diff_prog,
    'f_only_reg': f_only_reg}
