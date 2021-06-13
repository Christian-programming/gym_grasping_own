import gym
import numpy as np
import torch
from gym.spaces.box import Box
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.vec_normalize import VecNormalize as VecNormalize_
from baselines.common.vec_env.vec_normalize import DictVecNormalize as DictVecNormalize_


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:0] = 0
        return observation


class AddTimestep(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AddTimestep, self).__init__(env)
        self.observation_space = Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [self.observation_space.shape[0] + 1],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return np.concatenate((observation, [self.env._elapsed_steps]))


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class DictTransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(DictTransposeImage, self).__init__(env)
        img_obs_shape = self.observation_space.spaces['img'].shape
        self.observation_space.spaces['img'] = Box(
            self.observation_space.spaces['img'].low[0, 0, 0],
            self.observation_space.spaces['img'].high[0, 0, 0],
            [img_obs_shape[2], img_obs_shape[1], img_obs_shape[0]],
            dtype=self.observation_space.spaces['img'].dtype)

    def observation(self, observation):
        observation['img'] = observation['img'].transpose(2, 0, 1)
        return observation


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def reset_from_curriculum(self, data):
        obs = self.venv.reset_from_curriculum(data)
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.squeeze(1).cpu().numpy()
        self.venv.step_async(actions)

    def step_async_with_curriculum_reset(self, actions, data):
        actions = actions.squeeze(1).cpu().numpy()
        self.venv.step_async_with_curriculum_reset(actions, data)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class DictVecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(DictVecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = {'img': torch.from_numpy(obs['img']).float().to(self.device),
               'robot_state': torch.from_numpy(obs['robot_state']).float().to(self.device),
               'task_state': torch.from_numpy(obs['task_state']).float().to(self.device)}
        return obs

    def reset_from_curriculum(self, data):
        obs = self.venv.reset_from_curriculum(data)
        obs = {'img': torch.from_numpy(obs['img']).float().to(self.device),
               'robot_state': torch.from_numpy(obs['robot_state']).float().to(self.device),
               'task_state': torch.from_numpy(obs['task_state']).float().to(self.device)}
        return obs

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_async_with_curriculum_reset(self, actions, data):
        self.venv.step_async_with_curriculum_reset(actions, data)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = {'img': torch.from_numpy(obs['img']).float().to(self.device),
               'robot_state': torch.from_numpy(obs['robot_state']).float().to(self.device),
               'task_state': torch.from_numpy(obs['task_state']).float().to(self.device)}
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):

    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs):
        if self.ob_rms:
            if self.training:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class DictVecNormalize(DictVecNormalize_):

    def __init__(self, *args, **kwargs):
        super(DictVecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs):
        if isinstance(obs, dict):
            if self.ob_robot_rms:
                if self.training:
                    self.ob_robot_rms.update(obs['robot_state'])
                obs['robot_state'] = np.clip((obs['robot_state'] - self.ob_robot_rms.mean) /
                                             np.sqrt(self.ob_robot_rms.var + self.epsilon), -self.clipob,
                                             self.clipob)
                if self.training:
                    self.ob_task_rms.update(obs['task_state'])
                obs['task_state'] = np.clip(
                    (obs['task_state'] - self.ob_task_rms.mean) / np.sqrt(self.ob_task_rms.var + self.epsilon),
                    -self.clipob,
                    self.clipob)
                return obs
        else:
            if self.ob_rms:
                if self.training:
                    self.ob_rms.update(obs)
                obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
                return obs
            else:
                return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs,) + low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:]
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()


class DictVecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space.spaces['img']  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)
        self.device = device
        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs,) + low.shape).to(device)
        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        venv.observation_space.spaces['img'] = observation_space
        VecEnvWrapper.__init__(self, venv, observation_space=venv.observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = self.stacked_obs[:, self.shape_dim0:]
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs['img']
        obs['img'] = self.stacked_obs
        return obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        self.stacked_obs = torch.zeros(self.stacked_obs.shape).to(self.device)
        self.stacked_obs[:, -self.shape_dim0:] = obs['img']
        obs['img'] = self.stacked_obs
        return obs

    def reset_from_curriculum(self, data):
        obs = self.venv.reset_from_curriculum(data)
        self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        self.stacked_obs[:, -self.shape_dim0:] = obs['img']
        obs['img'] = self.stacked_obs
        return obs

    def close(self):
        self.venv.close()
