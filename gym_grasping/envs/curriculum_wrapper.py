"""
Set curriculum as wrapper around Env
Code is currently not in use, take curriculum wrapper from
"""
from collections import deque
import numpy as np
# from gym.utils import seeding
from baselines.common.vec_env import VecEnvWrapper
# from gym_grasping.envs.env_param_sampler import EnviromentParameterSampler
# from gym_grasping.envs.curricula import CURRICULA
# from gym_grasping.envs.curriculum_env import f_sr_prog


class CurriculumInfoWrapper(VecEnvWrapper):
    def __init__(self, venv, num_updates, num_update_steps, desired_rew_region, incr, tb_writer, num_processes):
        self.venv = venv
        super(CurriculumInfoWrapper, self).__init__(venv)
        self.num_updates = num_updates
        self.num_update_steps = num_update_steps
        self.num_processes = num_processes
        self.step_counter = 0
        self.update_counter = 0
        self.difficulty_cur = 0
        self.difficulty_reg = 0
        self.curr_episode_rewards = deque(maxlen=20)
        self.reg_episode_rewards = deque(maxlen=20)
        self.curr_success = deque(maxlen=20)
        self.reg_success = deque(maxlen=20)
        self.desired_rew_region = desired_rew_region
        self.incr = incr
        self.tb_writer = tb_writer
        self.num_regular_resets = 0
        self.num_resets = 0

    def update_difficulties(self):
        if len(self.curr_success) > 1:
            if np.mean(self.curr_success) > self.desired_rew_region[1]:
                self.difficulty_cur += self.incr
            elif np.mean(self.curr_success) < self.desired_rew_region[0]:
                self.difficulty_cur -= self.incr
            self.difficulty_cur = np.clip(self.difficulty_cur, 0, 1)
        if len(self.reg_success) > 1:
            if np.mean(self.reg_success) > self.desired_rew_region[1]:
                self.difficulty_reg += self.incr
            elif np.mean(self.reg_success) < self.desired_rew_region[0]:
                self.difficulty_reg -= self.incr
            self.difficulty_reg = np.clip(self.difficulty_reg, 0, 1)

    def create_data_dict(self):
        return {'update_step': self.update_counter,
                'num_updates': self.num_updates,
                'eprewmean': None,
                'curr_eprewmean': np.mean(self.curr_episode_rewards) if len(self.curr_episode_rewards) > 1 else 0,
                'eval_eprewmean': None,
                'reg_eprewmean': np.mean(self.reg_episode_rewards) if len(self.reg_episode_rewards) > 1 else 0,
                'curr_success_rate': np.mean(self.curr_success) if len(self.curr_success) > 1 else 0,
                'reg_success_rate': np.mean(self.reg_success) if len(self.reg_success) > 1 else 0,
                'eval_reg_eprewmean': None,
                'difficulty_cur': self.difficulty_cur,
                'difficulty_reg': self.difficulty_reg}

    def step(self, action):
        self.step_counter += 1
        if self.step_counter % self.num_update_steps == 0:
            self.update_counter += 1
            self.update_difficulties()
            self.write_tb_log()
            self.num_regular_resets = 0
            self.num_resets = 0
        data = self.create_data_dict()
        self.step_async_with_curriculum_reset(action, data)
        return self.step_wait()

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_async_with_curriculum_reset(self, actions, data):
        self.venv.step_async_with_curriculum_reset(actions, data)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        for i, info in enumerate(infos):
            if 'episode' in info.keys():
                if 'reset_info' in info.keys() and info['reset_info'] == 'curriculum':
                    self.curr_episode_rewards.append(info['episode']['r'])
                    self.curr_success.append(float(info['task_success']))
                    self.num_resets += 1
                elif 'reset_info' in info.keys() and info['reset_info'] == 'regular':
                    self.reg_episode_rewards.append(info['episode']['r'])
                    self.reg_success.append(float(info['task_success']))
                    self.num_resets += 1
                    self.num_regular_resets += 1
        return obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        return obs

    def reset_from_curriculum(self, data):
        obs = self.venv.reset_from_curriculum(data)
        return obs

    def write_tb_log(self):
        total_num_steps = (self.update_counter + 1) * self.num_processes * self.num_update_steps
        self.tb_writer.add_scalar("curr_success_rate", np.mean(self.curr_success) if len(self.curr_success) else 0, total_num_steps, total_num_steps)
        self.tb_writer.add_scalar("reg_success_rate", np.mean(self.reg_success) if len(self.reg_success) else 0, total_num_steps)
        self.tb_writer.add_scalar("difficulty_cur", self.difficulty_cur, total_num_steps)
        self.tb_writer.add_scalar("difficulty_reg", self.difficulty_reg, total_num_steps)

        if len(self.curr_episode_rewards) > 1:
            self.tb_writer.add_scalar("curr_eprewmean_steps", np.mean(self.curr_episode_rewards), total_num_steps)
            self.tb_writer.add_scalar("regular_resets_ratio", self.num_regular_resets / self.num_resets if self.num_resets > 0 else 0, total_num_steps)
        if len(self.reg_episode_rewards) > 1:
            self.tb_writer.add_scalar("reg_eprewmean_steps", np.mean(self.reg_episode_rewards), total_num_steps)

    def close(self):
        self.venv.close()
