"""
Policy to video, with plots
"""
import numpy as np
import gym
from a2c_ppo_acktr.play_model import Model, build_env, render_obs
from gym_grasping.scripts.viewer import Viewer


def run_policy():
    """
    Policy to video, with plots
    """
    env_id = "changing2_diff_reg_egl_stackVel-v0"
    env = gym.make(env_id)
    snapshot = "./policies/changing2_diff_reg_egl_stackVel-v0_2440.pt"

    env.seed(10)
    env = build_env(env, normalize_obs=False)
    model = Model(env, snapshot, deterministic=True)
    viewer = Viewer(video=True)
    obs = env.reset()
    done = False
    i = 0
    success = 0
    reward = 0
    ep_rews = 0
    while True:
        action, value = model.step(obs, done, return_value=True)
        obs, rew, done, info = env.step(action)
        value = value.numpy()[0, 0]

        viewer.step(obs, action, rew, value)
        render_obs(obs, sleep=10)
        done = done.any() if isinstance(done, np.ndarray) else done
        ep_rews += rew.cpu().flatten().numpy()[0]
        if done:
            if info[0]['task_success']:
                success += 1
            reward += ep_rews
            i += 1
            print("{} of {} successful, successrate: {}, avg reward: {} "
                  "reward: {}, ep_rew: {}".format(success, i, success / i,
                                                  reward / i, rew, ep_rews))

            # env resets itself when using build_env, which calls subprocvecenv
            ep_rews = 0


if __name__ == "__main__":
    run_policy()
