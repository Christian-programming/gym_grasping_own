"""
Calibrate the simulation dynamics by comparing recording from the real robot
to simulations where we can optimize parameters
"""
import math

import numpy as np
import torch
from gym import Wrapper
from scipy.optimize import differential_evolution

from gym_grasping.envs.robot_sim_env import RobotSimEnv
from gym_grasping.envs.iiwa_env import IIWAEnv
from robot_io.input_devices.space_mouse import SpaceMouse
from gym_grasping.utils import timeit

# TODO(lukas): old naming, remove once files have been renamed
# FILENAME_TEMPLATE = "./data/episodes/{}/episode_{}.npz"
FILENAME_TEMPLATE = "./data/simulation_dynamics_episode_{}.npz"


class Recorder(Wrapper):
    """
    Calibrate the simulation dynamics by comparing recording from the real
    robot to simulations where we can optimize parameters
    """

    def __init__(self, env, obs_type,
                 start_position_rel=(0, 0, 0.03, 0, 0, 0)):
        super(Recorder, self).__init__(env)
        self.initial_configuration = None
        self.start_position_rel = start_position_rel
        self.actions = []
        self.robot_state_observations = []
        self.obs_type = obs_type
        self.observation_space = env.observation_space
        assert self.obs_type in ['image_state', "img_color"]

    def step(self, action):
        '''step recording'''
        self.actions.append(action)
        observation, reward, done, info = self.env.step(action)
        self.robot_state_observations.append(info['robot_state_full'])
        if self.obs_type == "img_color":
            observation = observation['img']
        return observation, reward, done, info

    def reset(self):
        '''reset recording to beginning'''
        observation, info = self.env.reset(self.start_position_rel)
        self.initial_configuration = info['robot_state_full'][:4]
        if self.obs_type == "img_color":
            observation = observation['img']
        return observation

    def save(self, counter):
        '''save a recording'''
        np.savez("./data/simulation_dynamics_episode_{}.npz".format(counter),
                 self.initial_configuration, self.actions,
                 self.robot_state_observations)


def record_episode_3dmouse(mode, counter,
                           start_position_rel=(0, 0, 0.05, 0, 0, 0)):
    '''record a episode on real robot with the 3D mouse'''
    iiwa_env = IIWAEnv(act_type='continuous', freq=20,
                       obs_type='image_state_reduced', dv=0.01,
                       drot=0.1, use_impedance=True)
    iiwa_env = Recorder(env=iiwa_env, obs_type='image_state',
                        start_position_rel=start_position_rel)
    iiwa_env.reset()
    mouse = SpaceMouse('continuous')
    for _ in range(200):
        action = mouse.handle_mouse_events()
        mouse.clear_events()
        if mode == 0:
            # action[3] = 0
            action[4] = 1
        elif mode == 1:
            action[0] = 0
            action[1] = 0
            action[2] = 0
            action[4] = 1
        else:
            action[0] = 0
            action[1] = 0
            action[2] = 0
            action[3] = 0
        _, _, _, _ = iiwa_env.step(action)
    iiwa_env.save(counter=counter)


def record_episode_policy():
    '''record a learned policy on the real robot'''
    from a2c_ppo_acktr.play_model import Model, build_env, render_obs
    iiwa = IIWAEnv(act_type='continuous', freq=20,
                   obs_type='image_state_reduced', dv=0.01, drot=0.2,
                   use_impedance=True,
                   max_steps=1000, gripper_delay=0)
    iiwa = Recorder(env=iiwa, obs_type='image_state',
                    start_position_rel=(0, 0, 0.2, 0, 0, 0))
    snapshot = "/mnt/home_hermannl_vision/raid/hermannl/new_experiments/" \
               "white_block_model_05/" \
               "2019-08-24-06-37-48/save/ppo/stackVel_white-v0_2440.pt"

    env = build_env(iiwa, normalize_obs=False)
    model = Model(env, snapshot)

    obs = env.reset()
    done = False
    mouse = SpaceMouse('continuous')
    for _ in range(1000):
        action = model.step(obs, done)
        m_action = np.array(mouse.handle_mouse_events())
        mouse.clear_events()
        if not np.array_equal(m_action, np.array([0, 0, 0, 0, 1])):
            action = torch.Tensor(m_action).unsqueeze(0)
        obs, _, done, _ = env.step(action)
        render_obs(obs)
        # img = kinect.get_image()
        # img = cv2.resize(img, (640, 360))
        # cv2.imshow("win2", img[:, 140:500])
        # cv2.waitKey(1)

    iiwa.save(4)


def load_episode(i):
    '''load an episode from file'''
    data = np.load(FILENAME_TEMPLATE.format(i))
    initial_configuration = data["arr_0"]
    actions = data["arr_1"]
    observations = data["arr_2"]
    return initial_configuration, actions, observations


def rollout(env, initial_configuration, actions):
    '''rollout?'''
    # grasping env uses flange pos, we saved tcp pos, only works for gripper
    # facing down
    initial_configuration = initial_configuration.copy()
    initial_configuration[2] += 0.294
    env.reset(initial_position=(tuple(initial_configuration[:3]),
                                (-np.sqrt(2) / 2, -np.sqrt(2) / 2,
                                 0, 0)))
    observations = []
    for action in actions:
        obs, _, _, _ = env.step(action)
        # time.sleep(0.01)
        # we don't need task information
        observations.append(obs[:12])
    return np.array(observations)


def angle_between(angle_a, angle_b):
    '''map angles for some reason'''
    if -2.27 > angle_a:
        angle_a = 2 * np.pi + angle_a
    if -2.27 > angle_b:
        angle_b = 2 * np.pi + angle_b
    angle_a += 2.27
    angle_b += 2.27
    return abs(angle_a - angle_b)


def trajectory_similarity(real_obs, sim_obs):
    '''compute how similar trajectories are'''
    # consider tcp pos, EE angle and gripper opening width
    tcp_pos = np.sum(np.linalg.norm(real_obs[:, :3] - sim_obs[:, :3], axis=1))
    angles_zip = zip(real_obs[:, 3], sim_obs[:, 3])
    angles = np.sum([angle_between(a, b) for a, b in angles_zip])
    opening_width = np.sum(np.abs(real_obs[:, -1] - sim_obs[:, -1]))
    return tcp_pos, angles, opening_width


def test_function(x, episodes, env):
    '''test everythoing and return score'''
    act_dv = abs(x[0])
    act_drot = abs(x[1])
    joint_vel = abs(x[2])
    robot_delay = int(abs(x[3]))
    gripper_rot_vel = abs(x[4])
    max_rot_diff = abs(x[4])
    gripper_speed = abs(x[6])
    gripper_delay = int(abs(x[7]))
    frameskip = int(abs(x[8]))
    env.reset_simulation_params(act_dv, act_drot, joint_vel, robot_delay,
                                gripper_rot_vel,
                                max_rot_diff, gripper_speed,
                                gripper_delay, frameskip)
    score = 0
    for episode in episodes:
        initial_configuration, actions, real_obs = episode
        sim_obs = rollout(env, initial_configuration, actions)
        score += np.sum(trajectory_similarity(real_obs, sim_obs))
    # print(score)
    return score


def test_func_cartesian(x, episodes, env):
    '''teset cartesian position and return score'''
    d_x = abs(x[0])
    d_y = abs(x[1])
    d_z = abs(x[2])
    joint_vel = abs(x[3])
    frameskip = int(abs(x[4]))
    limit = abs(x[5]) > 0.5
    env.params.add_variable("robot_dv", tag="dyn", center=(d_x, d_y, d_z),
                            d=(0, 0, 0))
    env.params.add_variable("joint_vel", tag="dyn", center=joint_vel, d=0)
    env.params.add_variable("frameskip", tag="sim", center=frameskip, d=0,
                            f=int)
    env.robot.limit_movement_diff = bool(limit)
    env.params.init()
    score = 0
    for episode in episodes:
        initial_configuration, actions, real_obs = episode
        sim_obs = rollout(env, initial_configuration, actions)
        score += trajectory_similarity(real_obs, sim_obs)[0]
    print(score)
    return score


def test_func_rot(x, episodes, env):
    '''test rotation and return score'''
    act_drot = abs(x[0])
    gripper_rot_vel = abs(x[1])
    max_rot_diff = abs(x[2])
    env.reset_simulation_params(act_dv=0.005, act_drot=act_drot, joint_vel=5,
                                robot_delay=0,
                                gripper_rot_vel=gripper_rot_vel,
                                max_rot_diff=max_rot_diff, gripper_speed=5,
                                gripper_delay=0,
                                frameskip=4)
    score = 0
    for episode in episodes:
        initial_configuration, actions, real_obs = episode
        sim_obs = rollout(env, initial_configuration, actions)
        score += trajectory_similarity(real_obs, sim_obs)[1]
    # print(score)
    return score


def test_func_opening(x, episodes, env):
    '''these the gripper opening and return score'''
    gripper_speed = abs(x[0])
    gripper_delay = int(abs(x[1]))
    env.reset_simulation_params(act_dv=0.005, act_drot=0.2, joint_vel=5,
                                robot_delay=0,
                                gripper_rot_vel=5,
                                max_rot_diff=0.2, gripper_speed=gripper_speed,
                                gripper_delay=gripper_delay, frameskip=4)
    score = 0
    for episode in episodes:
        initial_configuration, actions, real_obs = episode
        sim_obs = rollout(env, initial_configuration, actions)
        score += trajectory_similarity(real_obs, sim_obs)[2]
    # print(score)
    return score


BOUNDS = [(0.0001, 0.01),  # act_dv
          (0.001, 0.1),  # act_drot
          (0.1, 10),  # joint_vel
          (0, 10),  # robot_delay
          (0.1, 10),  # gripper_rot_vel
          (0.05, 1),  # max_rot_diff
          (0.1, 10),  # gripper_speed
          (0, 10),  # gripper_delay
          (1, 10)]  # frameskip

BOUNDS_CART2 = [(0.0006, 0.0006),  # act_dv
                (0.0007, 0.0007),  # act_dv
                (0.001, 0.001),  # act_dv
                (1, 1),  # joint_vel
                (4, 4),  # frameskip
                (0, 0)]  # limit

BOUNDS_CART = [(0.0001, 0.05),  # act_dv
               (0.0001, 0.05),  # act_dv
               (0.0001, 0.05),  # act_dv
               (0.1, 10),  # joint_vel
               (1, 5),  # frameskip
               (0, 1)]  # limit

BOUNDS_ROT = [(0.001, 0.1),  # act_drot
              (0.1, 10),  # gripper_rot_vel
              (0.05, 1)]  # max_rot_diff

BOUNDS_OPENING = [(0.1, 10),  # gripper_speed
                  (0, 10)]  # gripper_delay


def optimize_simulation():
    '''run the differential evolution optimization on simulation'''
    env = RobotSimEnv(robot='kuka', task='stack', renderer='egl',
                      act_type='continuous',
                      initial_pose='calib',
                      max_steps=100000, obs_type='state')
    env.robot.navigate_in_cam_frame = False
    filenames = [4]
    episodes = []
    for i in filenames:
        episodes.append(load_episode(i))
    result = differential_evolution(func=test_func_cartesian,
                                    bounds=BOUNDS_CART,
                                    args=(episodes, env))
    print("cartesian")
    print(result)
    print("")
    #
    # env = RobotSimEnv(robot='kuka', task='block', renderer='tiny',
    #                   act_type='continuous',
    # initial_pose='calib',
    #                   max_steps=None, obs_type='state')
    # filenames = [0,1]
    # episodes = []
    # for i in filenames:
    #     episodes.append(load_episode(i, 1))
    # result = differential_evolution(func=test_func_rot, bounds=BOUNDS_ROT,
    #                                 args=(episodes, env))
    # print("rotation")
    # print(result)
    # print("")

    # env = RobotSimEnv(robot='kuka', task='block', renderer='tiny',
    #                   act_type='continuous',
    # initial_pose='calib',
    #                   max_steps=None, obs_type='state')
    # filenames = [0]
    # episodes = []
    # for i in filenames:
    #     episodes.append(load_episode(i, 2))
    # result = differential_evolution(func=test_func_cartesian,
    #                                 bounds=BOUNDS_OPENING,
    # args=(episodes, env))
    # print("opening")
    # print(result)
    # print("")


#
@timeit
def test_parameters(x):
    '''test given parameters, for opimizing'''
    env = RobotSimEnv(robot='kuka', task='block', renderer='tiny',
                      act_type='continuous',
                      initial_pose='calib',
                      max_steps=None, obs_type='state')
    filenames = range(10)
    episodes = []
    for i in filenames:
        episodes.append(load_episode(i))
    act_dv = abs(x[0])
    act_drot = abs(x[1])
    joint_vel = abs(x[2])
    robot_delay = int(abs(x[3]))
    gripper_rot_vel = abs(x[4])
    max_rot_diff = abs(x[4])
    gripper_speed = abs(x[6])
    gripper_delay = int(abs(x[7]))
    frameskip = int(abs(x[8]))
    env.reset_simulation_params(act_dv, act_drot, joint_vel, robot_delay,
                                gripper_rot_vel,
                                max_rot_diff, gripper_speed,
                                gripper_delay, frameskip)
    score = [0, 0, 0]
    for episode in episodes:
        initial_configuration, actions, real_obs = episode
        sim_obs = rollout(env, initial_configuration, actions)
        tcp_pos, angles, opening_width = trajectory_similarity(real_obs,
                                                               sim_obs)
        score[0] += tcp_pos
        score[1] += angles
        score[2] += opening_width
    print("tcp_pos: {}, angles: {}, opening_width: {}".format(*score))


def recalulate_tcp_pos(initial_configuration, actions, delta_v):
    '''reacalculate TPC positions?'''
    gripper_orn = math.pi / 4
    rot_mat = np.array([[np.cos(gripper_orn), -np.sin(gripper_orn), 0],
                        [np.sin(gripper_orn), np.cos(gripper_orn), 0],
                        [0, 0, 1]])

    positions = [initial_configuration[:3]]
    for action in actions:
        dxyz = action[:3] * delta_v
        dxyz = np.matmul(rot_mat, dxyz)
        positions.append(positions[-1] + dxyz)
    positions.pop(0)
    return np.array(positions)


def plot_trajectory(episode):
    '''plot whole trajectoies?'''
    env = RobotSimEnv(robot='kuka', task='stack', renderer='egl',
                      act_type='continuous',
                      initial_pose='calib',
                      max_steps=100000, obs_type='state')
    # env.reset_simulation_params(act_dv=3.37775125e-03, act_drot=0.2,
    #                             joint_vel=7.27652549,
    # robot_delay=0, gripper_rot_vel=5, max_rot_diff=0.2, gripper_speed=10,
    # gripper_delay=0,
    # frameskip=4)
    env.robot.navigate_in_cam_frame = False
    env.params.add_variable("robot_dv", tag="dyn",
                            center=(0.00528661, 0.010592, 0.00816933,),
                            d=(0, 0, 0))
    env.params.add_variable("joint_vel", tag="dyn", center=0.33276746, d=0)
    env.params.add_variable("frameskip", tag="sim", center=4, d=0, f=int)
    env.robot.limit_movement_diff = True
    env.params.init()
    initial_configuration, actions, obs = episode

    xyz = obs[:, :3]
    print(xyz.shape)
    fig = plt.figure()
    axis = fig.add_subplot(111)
    plt.subplots_adjust(bottom=0.3)
    axis.plot(xyz[:, 0], color="red")
    axis.plot(xyz[:, 1], color="green")
    axis.plot(xyz[:, 2], color="blue")

    # des_pos = recalulate_tcp_pos(initial_configuration, actions, 0.01)
    #
    # ax.plot(des_pos[:, 0])
    # ax.plot(des_pos[:, 1])
    # ax.plot(des_pos[:, 2])

    sim_obs = rollout(env, initial_configuration, actions)
    xyz = sim_obs[:, :3]
    print(xyz.shape)
    sim_x, = axis.plot(xyz[:, 0], "-.", color="red")
    sim_y, = axis.plot(xyz[:, 1], "-.", color="green")
    sim_z, = axis.plot(xyz[:, 2], "-.", color="blue")
    limit_mov = fig.add_axes([0.25, 0.25, 0.65, 0.03])
    d_x = fig.add_axes([0.25, 0.2, 0.65, 0.03])
    d_y = fig.add_axes([0.25, 0.15, 0.65, 0.03])
    d_z = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    vel = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    frameskip = fig.add_axes([0.25, 0, 0.65, 0.03])

    slimit = Slider(limit_mov, 'limit', 0, 1, valinit=1)
    sframeskip = Slider(frameskip, 'frameskip', 0, 5, valinit=4)
    sdx = Slider(d_x, 'dx', 0.0001, 0.02, valinit=0.00528661, valfmt='%1.4f')
    sdy = Slider(d_y, 'dy', 0.0001, 0.02, valinit=0.010592, valfmt='%1.4f')
    sdz = Slider(d_z, 'dz', 0.0001, 0.02, valinit=0.00816933, valfmt='%1.4f')
    svel = Slider(vel, 'vel', 0, 5, valinit=0.33276746)

    # sdelay = Slider(delay, 'delay', 0, 10, valinit=0)

    def update(val):
        # env.reset_simulation_params(act_dv=sdv.val, act_drot=0.2,
        # joint_vel=svel.val,
        # robot_delay=int(sdelay.val),
        # gripper_rot_vel=5, max_rot_diff=0.2, gripper_speed=10,
        # gripper_delay=0, frameskip=4)
        env.params.add_variable("robot_dv", tag="dyn",
                                center=(sdx.val, sdy.val, sdz.val),
                                d=(0, 0, 0))
        # env.params.add_variable("robot_drot", tag="dyn", .01)
        env.params.add_variable("joint_vel", tag="dyn", center=svel.val, d=0)
        env.params.add_variable("frameskip", tag="sim", center=sframeskip.val,
                                d=0, f=int)
        env.robot.limit_movement_diff = bool(slimit.val)
        env.params.init()
        sim_obs = rollout(env, initial_configuration, actions)
        X = np.arange(sim_obs.shape[0])
        sim_x.set_data(X, sim_obs[:, 0])
        sim_y.set_data(X, sim_obs[:, 1])
        sim_z.set_data(X, sim_obs[:, 2])
        fig.canvas.draw()
        print("loss: ", trajectory_similarity(obs, sim_obs)[0])
        print("dx", sdx.val)
        print("dy", sdy.val)
        print("dz", sdz.val)
        print("svel", svel.val)
        print('sframeskip', sframeskip.val)
        print('limit', slimit.val)
        print()

    sdx.on_changed(update)
    sdy.on_changed(update)
    sdz.on_changed(update)
    svel.on_changed(update)
    sframeskip.on_changed(update)
    slimit.on_changed(update)
    # sdelay.on_changed(update)

    plt.show()


def plot():
    '''plot whole trajectories?'''
    filenames = [1]
    episodes = []
    for i in filenames:
        episodes.append(load_episode(i, 0))
    plot_trajectory(episodes[0])


def plot_rotation():
    '''plot changes in rotation'''
    filenames = [1]
    episodes = []
    for i in filenames:
        episodes.append(load_episode(i, 1))
    episode = episodes[0]
    env = RobotSimEnv(robot='kuka', task='block', renderer='tiny',
                      act_type='continuous',
                      initial_pose='calib',
                      max_steps=None, obs_type='state')
    env.reset_simulation_params(act_dv=0.005, act_drot=0.2, joint_vel=5,
                                robot_delay=0,
                                gripper_rot_vel=5, max_rot_diff=0.2,
                                gripper_speed=10,
                                gripper_delay=0, frameskip=4)
    initial_configuration, actions, obs = episode

    xyz = obs[:, 3]
    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.plot(xyz, color="red")

    # des_pos = recalulate_tcp_pos(initial_configuration, actions, 0.01)
    #
    # ax.plot(des_pos[:, 0])
    # ax.plot(des_pos[:, 1])
    # ax.plot(des_pos[:, 2])

    sim_obs = rollout(env, initial_configuration, actions)
    xyz = sim_obs[:, 3]
    print(xyz)
    sim_x, = axis.plot(xyz, "-.")
    drot = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    vel = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    maxrot = fig.add_axes([0.25, 0, 0.65, 0.03])

    sdrot = Slider(drot, 'drot', 0.001, 0.1, valinit=0.02098671)
    svel = Slider(vel, 'vel', 0, 7, valinit=6.35354546, )
    smaxrot = Slider(maxrot, 'maxrot', 0, 0.5, valinit=0.09059702)

    def update(val):
        env.reset_simulation_params(act_dv=0.005, act_drot=sdrot.val,
                                    joint_vel=5, robot_delay=0,
                                    gripper_rot_vel=svel.val,
                                    max_rot_diff=smaxrot.val,
                                    gripper_speed=10, gripper_delay=0,
                                    frameskip=4)
        sim_obs = rollout(env, initial_configuration, actions)
        X = np.arange(sim_obs.shape[0])
        sim_x.set_data(X, sim_obs[:, 3])
        fig.canvas.draw()

    sdrot.on_changed(update)
    svel.on_changed(update)
    smaxrot.on_changed(update)

    plt.show()


def plot_opening():
    '''plot changes in gripper opening width'''
    filenames = [1]
    episodes = []
    for i in filenames:
        episodes.append(load_episode(i, 2))
    episode = episodes[0]
    env = RobotSimEnv(robot='kuka', task='block', renderer='tiny',
                      act_type='continuous',
                      initial_pose='calib',
                      max_steps=None, obs_type='state')
    env.reset_simulation_params(act_dv=0.005, act_drot=0.2, joint_vel=5,
                                robot_delay=0,
                                gripper_rot_vel=5, max_rot_diff=0.2,
                                gripper_speed=10,
                                gripper_delay=0, frameskip=4)
    initial_configuration, actions, obs = episode

    xyz = obs[:, -1]
    print(xyz)
    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.plot(xyz, color="red")

    # des_pos = recalulate_tcp_pos(initial_configuration, actions, 0.01)
    #
    # ax.plot(des_pos[:, 0])
    # ax.plot(des_pos[:, 1])
    # ax.plot(des_pos[:, 2])

    sim_obs = rollout(env, initial_configuration, actions)
    xyz = sim_obs[:, -1]
    sim_x, = axis.plot(xyz, "-.")
    f_s = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    vel = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    delay = fig.add_axes([0.25, 0, 0.65, 0.03])

    sfs = Slider(f_s, 'fs', 1, 8, valinit=4)
    svel = Slider(vel, 'vel', 0, 7, valinit=0.59996501)
    sdelay = Slider(delay, 'delay', 0, 16, valinit=9.0683065)

    def update(val):
        env.reset_simulation_params(act_dv=0.005, act_drot=0.02, joint_vel=5,
                                    robot_delay=0,
                                    gripper_rot_vel=5, max_rot_diff=0.2,
                                    gripper_speed=svel.val,
                                    gripper_delay=sdelay.val,
                                    frameskip=sfs.val)
        sim_obs = rollout(env, initial_configuration, actions)
        X = np.arange(sim_obs.shape[0])
        sim_x.set_data(X, sim_obs[:, -1])
        fig.canvas.draw()

    sfs.on_changed(update)
    svel.on_changed(update)
    sdelay.on_changed(update)

    plt.show()


if __name__ == "__main__":
    # record_episode_3dmouse(2, 1)
    # record_episode_policy()
    # optimize_simulation()

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    plot_trajectory(load_episode(4))
    # plot_rotation()
    # plot_opening()
    # test_parameters([2.83801620e-03, 9.36111822e-02, 1.18097406e+00,
    #                  3.29461239e+00,
    #                  0.2, 1.04882453e+00, 3.83012964e-01, 5.73886641e+00,
    #                  7.50828653e+00])
    # test_parameters([2.83418823e-03, 9.41566009e-02, 7.26331597e+00,
    #                  3.51585399e+00,
    #                  0.2, 1.06678573e+00, 3.82590451e+00, 4.74684988e+00,
    #                  7.63808358e+00])
    # test_parameters([3.91767936e-03, 9.66418198e-02, 7.57459391e+00,
    #                  1.56600222e+00,
    #                  0.2, 1.70485973e+00, 8.03399652e+00, 7.37814611e+00,
    #                  4.00000000e+00])
    # test_parameters([4.19111761e-03, 9.54600452e-02, 5.79523219e+00,
    #                  2.31415622e+00,
    #                 0.2, 1.77725733e+00, 1.15641211e+00, 7.01087585e+00,
    #                 4.00000000e+00])

    # test_parameters([4.07608062e-03, 1.21568152e-02, 5.00301992e-01,
    #                  5.69558066e+00,
    #    1.83523105e+00, 6.78729200e-01, 3.69842523e+00, 6.02269768e-01,
    #    4.00000000e+00])
    # test_parameters([0.007844  , 0.02429055, 7.12601244, 4.64836085,
    #                  3.81952119,
    #                  0.76451276, 6.92104135, 0.77408321, 2.47505408])

# calibration results of differential evolution:
# [5.58838448e-04, 1.22438135e-02, 4.65069626e+00, 6.91766910e+00]
# [5.62666538e-04, 1.22337583e-02, 3.80573074e+00, 6.09654457e+00]
