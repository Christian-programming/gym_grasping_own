import datetime
import os
import re
from os import listdir

import cv2
import numpy as np
from gym import Wrapper

from a2c_ppo_acktr.play_model import Model, build_env, render_obs

from gym_grasping.envs.iiwa_env import IIWAEnv
from robot_io.cams.kinect2 import Kinect2


class Recorder(Wrapper):
    def __init__(self, env, obs_type, save_dir, snapshot):
        super(Recorder, self).__init__(env)
        self.ep_counter = 0
        self.save_dir = save_dir
        try:
            os.mkdir(self.save_dir)
            with(open(os.path.join(self.save_dir, "info.txt"), 'w')) as f:
                f.write("time of recording: " + datetime.datetime.now().strftime(
                    "%Y-%m-%d-%H-%M-%S") + '\n')
                f.write("snapshot: " + snapshot)

        except FileExistsError:
            self.ep_counter = max(
                [int(re.findall(r'\d+', f)[0]) for f in listdir(save_dir) if f[-4:] == ".npz"]) + 1
        print(self.ep_counter)
        self.initial_configuration = None
        self.actions = []
        self.robot_state_observations = []
        self.robot_state_full = []
        self.img_obs = []
        # self.depth_imgs = []
        self.unscaled_imgs = []
        self.kinect_imgs = []
        self.obs_type = obs_type
        # self.observation_space = spaces.Box(low=0, high=255,
        #                                     shape=(84, 84, 3), dtype='uint8')
        assert self.obs_type in ['image_state', "img_color", "image_state_reduced"]
        self.kinect = Kinect2()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.actions.append(action)
        self.robot_state_observations.append(observation['robot_state'])
        self.robot_state_full.append(info['robot_state_full'])
        self.img_obs.append(observation['img'])
        if self.obs_type == "img_color":
            observation = observation['img']
        kinect_img = self.kinect.get_image()
        kinect_img = cv2.resize(kinect_img, (640, 360))[40:320, 180:460]
        self.kinect_imgs.append(kinect_img)
        # self.depth_imgs.append(info['depth'])
        self.unscaled_imgs.append(info['rgb_unscaled'])
        return observation, reward, done, info

    def reset(self):
        if len(self.img_obs) > 0:
            self.save()
            self.ep_counter += 1
        self.initial_configuration = None
        self.actions = []
        self.robot_state_observations = []
        self.robot_state_full = []
        self.img_obs = []
        self.kinect_imgs = []
        # self.depth_imgs = []
        self.unscaled_imgs = []

        observation = self.env.reset_at_random_position()
        self.initial_configuration = self.env._get_obs()[1]['robot_state_full'][:4]
        if self.obs_type == "img_color":
            observation = observation['img']
        return observation

    def save(self):
        path = os.path.join(self.save_dir, "episode_{}").format(self.ep_counter)
        np.savez(path,
                 initial_configuration=self.initial_configuration,
                 actions=self.actions,
                 steps=len(self.actions),
                 robot_state_observations=self.robot_state_observations,
                 robot_state_full=self.robot_state_full,
                 img_obs=self.img_obs,
                 kinect_imgs=self.kinect_imgs,
                 rgb_unscaled=self.unscaled_imgs)


def start_recording():
    iiwa = IIWAEnv(act_type='continuous', freq=20, obs_type='image_state_reduced', dv=0.01, drot=0.2,
                   use_impedance=True, max_steps=200)
    snapshot = "/mnt/home_hermannl/master_thesis/training_logs/2019-04-01-22-13-07/save/ppo/" \
               "changing2_diff_reg_egl_stackVel-v0_2440.pt"
    save_dir = '/media/kuka/Seagate Expansion Drive/kuka_recordings/stacking'
    recorder = Recorder(env=iiwa, obs_type='image_state_reduced', snapshot=snapshot,
                        save_dir=save_dir)
    env = build_env(recorder, normalize_obs=False)
    model = Model(env, snapshot)
    obs = env.reset()
    done = False
    while True:
        action = model.step(obs, done)
        obs, rew, done, info = env.step(action)
        render_obs(obs)


def load_episode(filename):
    data = np.load(filename)
    initial_configuration = data["initial_configuration"]
    actions = data["actions"]
    state_obs = data["robot_state_observations"]
    robot_state_full = data["robot_state_full"]
    img_obs = data["img_obs"]
    kinect_obs = data["kinect_imgs"]
    # depth = data['depth_imgs']
    rgb_unscaled = data['rgb_unscaled']
    steps = data["steps"]
    return initial_configuration, actions, state_obs, robot_state_full, img_obs, kinect_obs, \
        rgb_unscaled, steps


def load_episode_batch():
    folder = "/media/kuka/Seagate Expansion Drive/kuka_recordings/dr/2018-12-18-12-35-21"
    solved = 0
    for i in range(96):
        file = folder + "/episode_{}.npz".format(i)
        ep = load_episode(file)
        if ep[6]:
            solved += 1
    print(i / solved)


def show_episode(file, make_video=True):
    if not os.path.isfile(file):
        print("Missing file")
        return

    initial_configuration, actions, state_obs, robot_state_full, img_obs, kinect_obs, \
        rgb_unscaled, steps = load_episode(file)
    # print(initial_configuration)
    print(file)

    if make_video:
        video_name = file.replace(".npz", ".avi")
        video_name_ext = file.replace(".npz", "_ext.avi")

    for i in range(len(img_obs)):
        if make_video and i == 0:
            width, height = img_obs[i][:, :, ::-1].shape[:2]
            # fourcc = cv2.VideoWriter_fourcc(*'h264') # Be sure to use lower case
            fourcc = 0
            video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))
            width, height = kinect_obs[i].shape[:2]
            video_ext = cv2.VideoWriter(video_name_ext, fourcc, 30, (2 * width, height))

        if make_video:
            frame_robot = img_obs[i][:, :, ::-1]
            frame_ext = kinect_obs[i][:, :, :3]

            frame_robot_big = cv2.resize(frame_robot, (width, height))
            frame_cmb = np.concatenate((frame_ext, frame_robot_big), axis=1)
            # set_trace()

            video.write(frame_robot)
            video_ext.write(frame_cmb)

        # print(robot_state_full[i])
        # cv2.imshow("win2", rgb_unscaled[i, :,:,::-1])
        cv2.imshow("win", img_obs[i][:, :, ::-1])
        cv2.imshow("win2", kinect_obs[i][:, :, :3])
        # cv2.waitKey(0)

    if make_video:
        video.release()
        video_ext.release()
    print()


if __name__ == "__main__":
    # show_episode()
    # start_recording()

    for i in range(31):
        show_episode("/home/argusm/CLUSTER/full_episodes/episode_{}.npz".format(i))
