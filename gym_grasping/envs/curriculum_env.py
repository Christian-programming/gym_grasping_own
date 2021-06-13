"""
The currecnt structure of the code is:
pytorch/normalize -> CurriculumInfoWrapper  -> subproc -> CurriculumEnvLog

"""
from os import path
import numpy as np
import pybullet as p
from gym_grasping.envs.robot_sim_env import RobotSimEnv
from gym_grasping.envs.curricula import CURRICULA, RESTART_FUNCS


class DemonstrationEnv(RobotSimEnv):
    """
    Simplest form of Curriculum Env, randomly initialize rollouts
    """

    def __init__(self, *args, data_folder_path=None, task='stack',
                 max_curr_ep_len=None, **kwargs):
        super().__init__(*args, task=task, **kwargs)
        if data_folder_path is not None:
            data_path = path.join(path.dirname(__file__), "../curriculum",
                                  data_folder_path)
            self.data_folder_path = path.abspath(data_path)
            self.max_curr_ep_len = 150
            self.demo_with_dual_pivot_gripper = False
        else:
            if "stack" in task:
                data_path = path.join(path.dirname(__file__), "..",
                                      "curriculum/data_new")
                self.data_folder_path = path.abspath(data_path)
                self.max_curr_ep_len = 150
                self.demo_with_dual_pivot_gripper = True
            elif task in ('pick_n_stow', 'pick_n_stow_shaped'):
                self.data_folder_path = path.abspath(
                    path.join(path.dirname(__file__),
                              "..",
                              "curriculum/data_box_small"))
                self.max_curr_ep_len = 150
                self.demo_with_dual_pivot_gripper = True
            elif task == 'box_small_2obj':
                self.data_folder_path = path.abspath(
                    path.join(path.dirname(__file__),
                              "..",
                              "curriculum/data_box_small_2obj"))
                self.max_curr_ep_len = 150
                self.demo_with_dual_pivot_gripper = True
            else:
                print("Unknown task {}".format(task))
                raise ValueError
        if max_curr_ep_len is not None:
            self.max_curr_ep_len = max_curr_ep_len
        self.num_demonstrations = 10

    def get_simulator_state(self):
        '''get the simulator state'''
        closing_threshold = self.robot.gripper.closing_threshold
        gripper_open = self.robot.gripper.action_queue[0] > closing_threshold
        gripper_rotation = p.getJointState(self.robot.robot_uid,
                                           self.robot.active_gripper_index, physicsClientId=self.cid)[0]
        ee_pos = list(p.getLinkState(self.robot.robot_uid,
                                     self.robot.flange_index, physicsClientId=self.cid)[0])
        ee_pos[2] += 0.02
        workspace = self.robot.workspace
        return gripper_open, gripper_rotation, ee_pos, workspace

    def reset_from_file(self, episode, step):
        '''reset from a saved simulation state'''
        state_filename = "episode_{}/bullet_state_{}.bullet".format(episode,
                                                                    step)
        state_filename = path.join(self.data_folder_path, state_filename)
        p.restoreState(fileName=state_filename, physicsClientId=self.cid)
        if self.demo_with_dual_pivot_gripper:
            finger_link_id = self.robot.gripper.finger_link_ids[1]
            gripper_finger_pos = p.getJointState(self.robot.robot_uid,
                                                 finger_link_id, physicsClientId=self.cid)[0]
            p.resetJointState(self.robot.robot_uid,
                              self.robot.gripper.finger_link_ids[1],
                              gripper_finger_pos / 2, physicsClientId=self.cid)
        data_fn = path.join(self.data_folder_path,
                            "episode_{0}/episode_{0}.npz".format(episode))
        data = np.load(data_fn)
        gripper_open = True
        if not data['gripper_states'][step]:
            self.robot.gripper.reset(gripper_open=False)
            gripper_open = False
        self.robot.desired_ee_angle = data['ee_angles'][step]
        self.robot.desired_ee_pos = data['ee_positions'][step]
        self.robot.workspace = data['workspace']
        self._task.reset_from_curriculum()
        return gripper_open

    # data between 0 (difficult) and 1 (easy)
    def reset(self, data=None):
        '''do a reset'''
        if data is None:
            self.eval = True
            obs = super().reset()
            return obs
        self.eval = False
        obs = super().reset()
        episode = self.np_random.randint(self.num_demonstrations)
        step = self.np_random.randint(self.max_curr_ep_len)
        self.reset_from_file(episode, step)
        p.stepSimulation(physicsClientId=self.cid)
        obs = self._get_obs()
        return obs


class CurriculumEnvLinear(DemonstrationEnv):
    """
    Linearly increase difficulty according to training progress and use regular
    episode starts with increasing probability
    """

    def __init__(self, *args, use_regular_starts=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.std = 10
        self.use_regular_starts = use_regular_starts

    # data between 0 (difficult) and 1 (easy)
    def reset(self, data=None):
        obs = super().reset()
        if data is None:
            return obs
        episode = self.np_random.randint(self.num_demonstrations)
        difficulty = data['update_step'] / data['num_updates']
        # use regular restarts with increasing probability
        if self.use_regular_starts and self.np_random.rand() < difficulty:
            return obs
        mu = (self.max_curr_ep_len - self.std / 2) * (1 - difficulty)
        uniform = self.np_random.uniform(low=mu - self.std, high=mu + self.std)
        step = int(np.clip(uniform, 0, self.max_curr_ep_len))
        self.reset_from_file(episode, step)
        p.stepSimulation(physicsClientId=self.cid)
        obs = self._get_obs()
        return obs


class AdaptiveCurriculumEnv(DemonstrationEnv):
    """
    Adaptively increase or decrease difficulty according to training rewards

    """

    def __init__(self, *args, curr="easy", std=10, use_regular_starts=False,
                 adaptive_task_difficulty=False, reg_start_func='f_eval_prog',
                 diff="diff_cur_reg", **kwargs):
        self.adaptive_task_difficulty = adaptive_task_difficulty
        # needed for reset
        self.curr = CURRICULA[curr]()
        self.reg_start_probabilty = RESTART_FUNCS[reg_start_func]
        super().__init__(*args, **kwargs)

        self.std = std
        self.use_regular_starts = use_regular_starts
        self.diff = diff
        assert diff in ["diff_cur", "diff_cur_reg", "diff_linear"]
        self.episode_info = {}

    def reset(self, data=None):
        '''data between 0 (difficult) and 1 (easy)'''
        self.episode_info = {}
        if data is None:
            self.eval = True
            # for eval
            if self.adaptive_task_difficulty:
                self.curr.set_difficulty(self.params, 1)
            obs = super().reset()
            return obs
        self.eval = False
        progress = data['update_step'] / data['num_updates']
        eval_eprewmean = data['eval_eprewmean']
        curr_eprewmean = data['curr_eprewmean']
        difficulty_cur = data['difficulty_cur']
        difficulty_reg = data['difficulty_reg']
        curr_sr = data['curr_success_rate']
        reg_sr = data['reg_success_rate']
        if self.diff == "diff_cur":
            difficulty_reg = difficulty_cur
        elif self.diff == "diff_linear":
            difficulty_cur = difficulty_reg = progress
        reg_eprewmean = data['reg_eprewmean']
        eval_reg_eprewmean = data['eval_reg_eprewmean']

        reg_start_p = self.reg_start_probabilty(difficulty_cur, progress,
                                                eval_eprewmean, curr_eprewmean,
                                                reg_eprewmean,
                                                eval_reg_eprewmean, curr_sr,
                                                reg_sr)
        do_regular_restart = self.use_regular_starts and \
            self.np_random.rand() < reg_start_p

        # do_regular_start=None  -> eval reset
        # do_regular_start=True  -> train regular
        # do_regular_start=False -> train curriculum
        if self.adaptive_task_difficulty:
            if do_regular_restart:
                self.curr.set_difficulty(self.params, difficulty_reg,
                                         do_regular_restart)
            else:
                self.curr.set_difficulty(self.params, difficulty_cur,
                                         do_regular_restart)
        obs = super().reset()
        self.episode_info = self.params.get_curriculum_info()
        self.episode_info['step'] = None
        # use regular restarts with increasing probability
        if do_regular_restart:
            self._info['reset_info'] = 'regular'
            return obs
        # code for curriculum reset
        episode = self.np_random.randint(self.num_demonstrations)
        mu = (self.max_curr_ep_len - self.std / 2) * (1 - difficulty_cur)
        uniform = self.np_random.uniform(low=mu - self.std, high=mu + self.std)
        step = int(np.clip(uniform, 0, self.max_curr_ep_len))
        self.reset_from_file(episode, step)
        p.stepSimulation(physicsClientId=self.cid)
        obs = self._get_obs()
        # adapt episode length for curriculum resets
        self._max_steps = int(self._default_max_steps - step +
                              (self._default_max_steps / 3) *
                              (step / self._default_max_steps))
        self._info['reset_info'] = 'curriculum'
        self.episode_info['step'] = step
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if done and 'reset_info' in info:
            info['episode_info'] = {'ep_length': self._ep_step_counter,
                                    'reset_type': info['reset_info'],
                                    **self.episode_info}
        return obs, reward, done, info


def test_curriculum_env():
    '''tese the curriculum env'''
    from robot_io.input_devices.space_mouse import SpaceMouse
    import cv2
    env = DemonstrationEnv(task='box_small', renderer='debug',
                           act_type='continuous', initial_pose='close',
                           max_steps=1e3, obs_type='image', param_randomize=False,
                           camera_pos="new_mount")
    mouse = SpaceMouse()
    while 1:
        for _ in range(20):
            action = mouse.handle_mouse_events()
            mouse.clear_events()
            obs, _, done, _ = env.step(action)
            img = cv2.resize(obs[:, :, ::-1], (300, 300))
            cv2.imshow("win", img)
            cv2.waitKey(1)
            if done:
                env.reset(True)
        env.reset(True)


if __name__ == "__main__":
    test_curriculum_env()
