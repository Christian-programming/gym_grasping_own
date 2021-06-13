"""
compare how policys change as a result of parameter changes.
"""
import cv2
import matplotlib.colors
import numpy as np
from PIL import Image, ImageEnhance
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.ppo2.ppo2_eval_combi import get_ppo2_runner

from gym_grasping.envs.robot_sim_env import RobotSimEnv

BOUNDS = [(0, 2),  # bright
          (0, 2),  # con
          (0, 2),  # col
          (0, 2),  # sharp
          (0, 20),  # blur
          (-0.1, 0.1)]  # hue


def load_episode(filename):
    data = np.load(filename)
    initial_configuration = data["arr_0"]
    actions = data["arr_1"]
    state_obs = data["arr_2"]
    img_obs = data["arr_3"]
    return initial_configuration, actions, state_obs, img_obs


def policy_act(model, obs):
    action, _, _, _ = model.step([obs], None, False)
    action = np.clip(action[0], -1, 1)
    return action


def transform_img(img, params):
    bright = params[0]
    con = params[1]
    col = params[2]
    sharp = params[3]
    blur = params[4]
    hue = params[5]

    img = Image.fromarray(img)
    contrast = ImageEnhance.Contrast(img)
    img = contrast.enhance(con)
    brightness = ImageEnhance.Brightness(img)
    img = brightness.enhance(bright)
    color = ImageEnhance.Color(img)
    img = color.enhance(col)
    for _ in range(5):
        sharpness = ImageEnhance.Sharpness(img)
        img = sharpness.enhance(sharp)
    blur_fac = int(blur) * 2 + 1
    img = cv2.GaussianBlur(np.array(img), ksize=(blur_fac, blur_fac), sigmaX=0)
    img = img / 255
    img_hsv = matplotlib.colors.rgb_to_hsv(img)
    img_hsv[:, :, 0] += hue
    img = matplotlib.colors.hsv_to_rgb(img_hsv)
    img = np.array(img * 255, dtype=np.uint8)

    return img


def use_policy(img_obs, params, model):
    actions = []
    for obs in img_obs:
        img = transform_img(obs, params)
        # cv2.imshow("win1", obs)
        # cv2.imshow("win2", rgb)
        # cv2.waitKey(0)
        action = policy_act(model, img)
        print(action)
        actions.append(action)
    return actions


def action_similarity(real_actions, pred_actions):
    real_actions = np.array(real_actions)
    pred_actions = np.array(pred_actions)
    score = np.sum(np.linalg.norm(pred_actions - real_actions, axis=1))
    return score


def test_func(x, model, actions, obs):
    pred_actions = use_policy(obs, x, model)
    return action_similarity(actions, pred_actions)


def test_parameters(params, model, actions, obs):
    pred_actions = use_policy(obs, params, model)
    print(action_similarity(actions, pred_actions))


def test_random(actions):
    random_actions = np.random.random((len(actions), 5)) * 2 - 1
    print(action_similarity(actions, random_actions))


def test_random_discrete(actions):
    random_actions = np.random.randint(-1, 2, size=(len(actions), 5))
    print(action_similarity(actions, random_actions))


def main():
    snapshot_fn = "/home/kuka/Downloads/02000(2)"
    env = RobotSimEnv(robot='kuka', task='block', initial_pose='close', act_type='continuous',
                      renderer='tiny',
                      act_dv=0.001, obs_type='image')
    env_v = DummyVecEnv([lambda: env])
    model = get_ppo2_runner(env_v, nsteps=1, load_path=snapshot_fn).model
    filename = "/home/kuka/lang/robot/gym_grasping/gym_grasping/recordings/data/episode_0.npz"

    _, actions, _, img_obs = load_episode(filename)
    # result = differential_evolution(func=test_func, bounds=BOUNDS, args=(model, actions, img_obs))
    # print(result)
    params = [0.02416125, 0.95295959, 1.93125142, 1.48887453, 3.07533129, 0.07888074]
    params2 = [0.02428529, 0.92042836, 1.1155627, 1.0774323, 3.88660779, 0.08824335]
    params3 = [0.03730358, 1.70839662, 1.57078595, 1.79412154, 10.06657854, 0.09279722]
    params4 = [0.02880297, 1.26974988, 1.17036981, 0.03405814, 18.60460761, 0.09640106]
    standard_params = [1, 1, 1, 1, 0, 0]
    hue = [1, 1, 1, 1, 0, 0.09]
    test_parameters(params, model, actions, img_obs)
    test_parameters(params2, model, actions, img_obs)
    test_parameters(params3, model, actions, img_obs)
    test_parameters(params4, model, actions, img_obs)
    test_parameters(hue, model, actions, img_obs)
    test_parameters(standard_params, model, actions, img_obs)
    test_random(actions)
    test_random_discrete(actions)


if __name__ == "__main__":
    main()
