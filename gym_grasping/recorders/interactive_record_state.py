"""
Save robot states wile interacting with it, this is for development.
"""
import os
import json
import datetime
from PIL import Image

import pybullet as p
from gym import Wrapper
from gym_grasping.envs.robot_sim_env import RobotSimEnv


class SaveWrapper(Wrapper):
    '''Wrapper to save env transitions while interacting with it.'''
    def __init__(self, env, save_dir=None):
        Wrapper.__init__(self, env)
        self.ep_count = 0
        if save_dir is None:
            str_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
            save_dir = "{}".format(str_time)
        self.save_dir = save_dir
        self.filename = os.path.join(save_dir, "info.json")
        self.image_fn = '{0:03d}.png'
        # make dir and file
        os.makedirs(save_dir, exist_ok=True)
        with open(self.filename, 'a'):
            os.utime(self.filename)

    def step(self, action):
        '''follow a step'''
        obs, reward, done, info = self.env.step(action)
        if done:
            self.save(obs, reward, done, info, self.ep_count)
            self.ep_count += 1
        return obs, reward, done, info

    def reset(self):
        '''reset the class'''
        obs = self.env.reset()
        return obs

    def save(self, obs, reward, done, info, ep_count):
        '''save data on done'''
        if done:
            # save last image as PNG
            image_fn = self.image_fn.format(ep_count)
            img = Image.fromarray(obs)
            img.save(os.path.join(self.save_dir, image_fn))
            # update info dict
            info["reward"] = reward
            info["done"] = done
            info["obs"] = image_fn
            # pickle and save
            info_str = json.dumps(info)
            with open(self.filename, 'a') as file_obj:
                file_obj.write(info_str+'\n')


def interactive_play(task="block"):
    """
    Control the robot interactively by controlling sliders in the debug viewer.
    Be carefull not to dof sliders while over the robot becasue mouse actions
    go through panel.
    """
    env = RobotSimEnv(task=task, renderer='debug', act_type='continuous',
                      max_steps=None, initial_pose="close")
    policy = env._task.get_policy(mode='play')(env)

    # was this for no jerks at end of policy?
    policy_defaults = policy.get_defaults()

    control_names = env.robot.control_names
    if policy_defaults is not None:
        defaults = policy.get_defaults()
    else:
        defaults = [0, ]*len(control_names)

    motors_ids = []
    for ctrl_name, delta_v in zip(control_names, defaults):
        motors_ids.append(env.p.addUserDebugParameter(ctrl_name, -1, 1, delta_v))
    motors_ids.append(env.p.addUserDebugParameter("debug", -1, 1, 0))

    env = SaveWrapper(env)

    done = False
    while not done:
        action = []
        for motor_id in motors_ids:
            action.append(p.readUserDebugParameter(motor_id))
        debug = action[-1] > .5
        action = policy.act(env, action)
        _, reward, done, _ = env.step(action)
        print("reward:", reward)
        if debug == 1.0:
            env.reset()
        if reward == 1.0:
            done = False
            env.reset()
    print(reward)


if __name__ == "__main__":
    interactive_play()
