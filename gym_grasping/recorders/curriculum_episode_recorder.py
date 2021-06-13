"""
Record episodes for using in a curriculum
"""
import argparse
import os
import time

import cv2
import numpy as np
import pybullet as p
from gym import Wrapper
from gym import spaces

from gym_grasping.envs.curriculum_env import DemonstrationEnv
from gym_grasping.envs.robot_sim_env import RobotSimEnv
from robot_io.input_devices.space_mouse import SpaceMouse


class Recorder(Wrapper):
    '''Record episodes for using in a curriculum'''
    def __init__(self, env, save_dir="./tmp_record", episode=0):
        super(Recorder, self).__init__(env)
        self.episode = episode

        self.save_dir = os.path.join(save_dir, "episode_{}".format(episode))
        os.makedirs(self.save_dir, exist_ok=True)

        self.initial_configuration = None
        self.actions = []
        self.image_observations = []
        self.robot_state_observations = []
        self.task_state_observations = []
        self.img_obs = []
        self.step_counter = 0
        self.gripper_states = []
        self.ee_angles = []
        self.ee_positions = []
        self.workspace = None
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(84, 84, 3), dtype='uint8')

    def step(self, action):
        '''step recorder'''
        observation, reward, done, info = self.env.step(action)
        self.actions.append(action)
        self.image_observations.append(observation['img'].copy())
        self.robot_state_observations.append(observation['robot_state'])
        self.task_state_observations.append(observation['task_state'])
        self.img_obs.append(observation['img'])
        bullet_fn = os.path.join(self.save_dir, "bullet_state_{}.bullet".format(self.step_counter))
        p.saveBullet(bulletFileName=bullet_fn)
        gripper_state, ee_angle, ee_pos, self.workspace = self.env.get_simulator_state()
        self.gripper_states.append(gripper_state)
        self.ee_angles.append(ee_angle)
        self.ee_positions.append(ee_pos)
        self.step_counter += 1
        return observation, reward, done, info

    def reset(self):
        '''reset class'''
        observation = self.env.reset()
        self.initial_configuration = observation['robot_state'][:4]
        return observation

    def save(self, save_name=None):
        '''save episode data'''
        if save_name is None:
            save_name = self.episode
        np.savez(self.save_dir + "/episode_{}.npz".format(save_name),
                 gripper_states=self.gripper_states,
                 ee_angles=self.ee_angles,
                 ee_positions=self.ee_positions,
                 workspace=self.workspace)
        # np.savez(self.save_dir + "/episode_{}_img.npz".format(save_name),
        #          img=self.image_observations)


def replay_episode(task="stack", episode=0):
    '''replay episode from demo'''
    env = DemonstrationEnv(robot='kuka', task=task, renderer='debug',
                           act_type='continuous', initial_pose='close',
                           max_steps=200, obs_type='image_state')
    env.reset()
    for i in range(200):
        env.reset_from_file(episode, i)
        env.render()
        time.sleep(0.01)


def record_episode_3dmouse(task="stack", episode=0):
    '''recurd using the 3D mouse'''
    env = DemonstrationEnv(robot='kuka', task=task, renderer='tiny',
                           act_type='continuous', initial_pose='close',
                           max_steps=200, obs_type='image_state',
                           img_size=(256, 256), param_randomize=False)
    env.seed(episode)
    env = Recorder(env=env, episode=episode)
    # env.reset()
    mouse = SpaceMouse('continuous')

    for i in range(200):
        action = mouse.handle_mouse_events()
        mouse.clear_events()
        obs, _, done, _ = env.step(action)
        img = cv2.resize(obs['img'][:, :, ::-1], (300, 300))
        cv2.imshow("win", img)
        cv2.waitKey(10)
    env.save()
    print()
    print("Number of steps recorded", i, "<<<<<<<")


# TODO(lukas): is this needed
def reset_simulation():
    '''dont know if this is needed'''
    env = RobotSimEnv(robot='kuka', task='stack', renderer='debug', act_type='continuous',
                      initial_pose='close',
                      max_steps=200, obs_type='image_state')
    mouse = SpaceMouse('continuous')
    while 1:
        gripper_open = True
        env.reset(data=True)
        mouse._gripper_state = int(gripper_open)
        for _ in range(1):
            action = mouse.handle_mouse_events()
            mouse.clear_events()
            _, _, done, _ = env.step(action)
            time.sleep(0.01)
            if done:
                break


def main():
    '''run the recorder'''
    parser = argparse.ArgumentParser()

    # what to do
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-p", "--play", dest='mode',
                      action="store_const", const="play", help="play demonstration")
    mode.add_argument("-r", "--record", dest='mode',
                      action="store_const", const="record", help="record demonstration")
    parser.set_defaults(mode='play')

    parser.add_argument("--task", type=str, default="bolt",
                        help="task")

    # where to look
    parser.add_argument("--dir", type=str, default="./tmp_record/",
                        help="environoment")

    parser.add_argument("--num", type=int, default=1,
                        help="number of episodes")

    # what to record
    # record both of these by default
    # parser.add_argument("--recrod_images", type=bool, default=True,
    #                    help="record images")
    # parser.add_argument("--record_snapshots", type=bool, default=True,
    #                    help="record simulation snapshots")
    args = parser.parse_args()

    for i in range(args.num):
        if args.mode == "play":
            replay_episode(task=args.task, episode=i)

        elif args.mode == "record":
            record_episode_3dmouse(task=args.task, episode=i)
        else:
            raise ValueError("Unknown mode", args.mode)
        # reset_simulation()


if __name__ == "__main__":
    main()
