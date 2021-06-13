"""
Testing file for development, to experiment with evironments.
This is max's default playing around with stuff file.
"""
import time
import argparse
import traceback
from pdb import set_trace

import gym
import numpy as np
import pybullet as p
from matplotlib import pyplot as plt
from gym_grasping.envs.robot_sim_env import RobotSimEnv


def interactive_play(env=None, task_name="pick_n_place", mouse=True,
                     show=False):
    """
    Control the robot interactively by controlling sliders in the debug viewer.
    Note:
    Be carefull not to click on sliders while over the robot becasue mouse actions
    go through the rendered panel.
    """
    print(task_name)
    if show:
        import cv2

    if mouse:
        from robot_io.input_devices.space_mouse import SpaceMouse
        mouse = SpaceMouse(act_type='continuous')
        print("SpaceMouse found.")

    # default variables
    renderer = "debug"
    robot = "kuka"
    control = "relative"
    param_info = None

    if env is not None:
        # don't create a new env
        pass
    else:
        if task_name == 'stack':
            # param_fn = "./normal_visual_domain_randomization_range_v1.json"
            # with open(param_fn, "r") as param_obj:
            #    import json
            #    param_info = json.load(param_obj)
            pass
        elif task_name in ('pick_n_place', 'CAD', 'vacuum'):
            robot = "suction"
            control = "absolute"

        # create a new env
        env = RobotSimEnv(task=task_name, robot=robot, renderer=renderer, control=control,
                          param_info=param_info, show_workspace=True,
                          max_steps=1000)

    control_names = env.robot.control_names
    defaults = [0, ]*len(control_names)
    policy = True if hasattr(env._task, "policy") else False

    motors_ids = []
    for ctrl_name, delta_v in zip(control_names, defaults):
        motors_ids.append(p.addUserDebugParameter(ctrl_name, -1, 1, delta_v))
    motors_ids.append(p.addUserDebugParameter("debug", -1, 1, 0))

    debug = False
    done = False
    counter = 0
    while 1:
        if policy:
            action, policy_done = env._task.policy(env, None)

        if mouse:
            action = mouse.handle_mouse_events()
            mouse.clear_events()
        else:
            action = []
            for motor_id in motors_ids:
                action.append(p.readUserDebugParameter(motor_id))
            debug = action[-1] > .5
            action = action[:-1]

        # action = policy.act(env, action)
        state, reward, done, _ = env.step(action)
        # print("reward:", reward)
        if done or debug:
            env.reset()

        if show:
            img = state[:, :, ::-1]
            cv2.imshow('window', cv2.resize(img, (300, 300)))
            cv2.waitKey(10)

        counter += 1

    print(reward)


def random_policy(env, task_name):
    """
    Execute the the random policy.
    """
    # import cv2
    if env is None:
        env = RobotSimEnv(act_type='continuous', task=task_name,
                          renderer='egl', initial_pose='close')
        # env_id = gym.make('KukaBoltMD-v0')
        # env = gym.make(env_id)

    fig, axis = plt.subplots()
    axis.set_axis_off()
    handle = axis.imshow(np.zeros((84, 84, 3), dtype=np.uint8))
    fig.canvas.draw()
    plt.show(block=False)

    env.reset()
    done = False
    while 1:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        state = state
        handle.set_data(state)
        handle.set_clim(vmin=state.min(), vmax=state.max())
        fig.canvas.draw()
        # cv2.imshow('window', cv2.resize(state[:, :, ::-1], (300, 300)))
        # cv2.waitKey(1)
        time.sleep(.02)
        if done:
            env.reset()
    print(reward)


def profile_env(env, task_name='block'):
    """
    Test success rate of random policy, indication of learnability.
    """
    if env is None:
        env = RobotSimEnv(task=task_name, act_type='continuous',
                          initial_pose='close',
                          renderer='tiny')
    else:
        env = gym.make(env)

    log_id = p.startStateLogging(
        p.STATE_LOGGING_PROFILE_TIMINGS, "renderTimings")

    success_count = 0
    ep_count = 0
    start_time = time.time()
    env_done = False
    for iteration in range(2000):
        action = env.action_space.sample()
        state, reward, env_done, info = env.step(action)
        if env_done and reward > 0:
            success_count += 1

        if env_done:
            env.reset()
            ep_count += 1

        if env_done and "clear" in info and "first" in info:
            plt.imshow(state)
            plt.show()

        if iteration % 100 == 0 and iteration > 0:
            print("FPS: ", 100 / (time.time() - start_time))
            start_time = time.time()

    p.stopStateLogging(log_id)

    print("success: ", success_count)
    print("iterations: ", iteration+1)
    print("episodes:", ep_count)
    print("success rate iteration:", success_count/(iteration+1))
    print("success rate episode: ", success_count/ep_count)


def test_env(env, task_name):
    """
    Test envs for errors.
    """
    if env is None:
        env = RobotSimEnv(task=task_name,
                          act_type='continuous',
                          max_steps=None, initial_pose='close')
    tested_task = {task: " not tested" for task in RobotSimEnv.tasks}
    renderer = 'egl'
    # renderer = 'debug'
    for task in RobotSimEnv.tasks:
        # for task in ["stack"]:
        # test all task
        print("testing task: {}".format(task))
        try:
            env = RobotSimEnv(robot='kuka', task=task, renderer=renderer,
                              act_type='continuous', initial_pose='close',
                              obs_type='image', act_dv=0.001)
            for i in range(1000):
                if i % 100 == 0:
                    env.reset()
                action = env.action_space.sample()
                _ = env.step(action)
            del env
        except Exception as error:
            print("TASK {} ERROR: \n{}".format(
                task, traceback.format_exc()))
            tested_task[task] = "failed"
            error  # for PEP8?

    for task, test_result in tested_task.items():
        print("task {:<15} | result {:<5}".format(task, test_result))


def sample_env(env, task_name):
    """
    Sample Enviroment Parameters.
    """
    import cv2

    if env is None:
        env = RobotSimEnv(act_type='continuous', task=task_name,
                          renderer='egl', initial_pose='close')
        # env_id = "changing2_diff_reg_egl_stackVel-v0"
        # env = gym.make(env_id)

    # env = gym.make('KukaBoltMD-v0')
    policy = env._task.get_policy(mode='random')(env)

    env.reset()
    done = False
    while 1:
        action = policy.act(env)
        state, reward, done, _ = env.step(action)
        state = state['img']
        cv2.imshow('window', cv2.resize(state[:, :, ::-1], (300, 300)))
        cv2.waitKey(1)
        time.sleep(.02)
        if done:
            env.reset()
    print(reward)


def params_env(env, task_name):
    '''print a given envs parameters'''
    if env is None:
        env = RobotSimEnv(task=task_name, act_type='continuous',
                          renderer='tiny', initial_pose='close')
    params = env.params.variables
    print("{:<20} {:<20} {:<20}".format('Name', 'Center', 'Delta'))
    for par_name, par_data in params.items():
        if "center" in par_data:
            c_str = "{:<20} {:<20} {:<20}"\
              .format(par_name, str(par_data["center"]), str(par_data["d"]))
            print(c_str)
        elif "ll" in par_data and "ul" in par_data:
            center = (.5*np.array(par_data["ul"]) +
                      .5*np.array(par_data["ll"])).tolist()
            delta = (np.array(par_data["ul"]) -
                     np.array(par_data["ll"])).tolist()
            print("{:<20} {:<20} {:<20}".format(par_name+"*", str(center),
                                                str(delta)))
        else:
            print("{:<20} {:<20} {:<20}".format(par_name, "???", "???"))
    print()
    set_trace()


def main():
    '''parse args an run'''
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-p", "--play", dest='mode', action="store_const",
                      const="play", help="interactive mode")
    mode.add_argument("-r", "--random", dest='mode', action="store_const",
                      const="random", help="random policy")
    mode.add_argument("-e", "--enjoy", dest='mode', action="store_const",
                      const="enjoy", help="learned policy")
    mode.add_argument("-o", "--profile", dest='mode', action="store_const",
                      const="profile", help="test performance")
    mode.add_argument("-t", "--test", dest='mode', action="store_const",
                      const="test", help="text execution")
    mode.add_argument("-s", "--sampling", dest='mode', action="store_const",
                      const="sample", help="view param sampling")
    mode.add_argument("--params", dest='mode', action="store_const",
                      const="params", help="list env params")
    parser.set_defaults(mode='play')

    parser.add_argument("--env", type=str, default=None,
                        help="environoment")
    parser.add_argument("--task", type=str, default="stack",
                        help="task name")

    args = parser.parse_args()
    env = args.env
    if env is not None:
        env = gym.make(env)
    task = args.task

    if args.mode == "play":
        interactive_play(env=env, task_name=task)
    elif args.mode == "random":
        random_policy(env=env, task_name=task)
    # elif args.mode == "enjoy":
    #    enjoy_policy(env=env, task_name=task)
    elif args.mode == "profile":
        profile_env(env=env, task_name=task)
    elif args.mode == "test":
        test_env(env=env, task_name=task)
    elif args.mode == "sample":
        sample_env(env=env, task_name=task)
    elif args.mode == "params":
        params_env(env=env, task_name=task)
    else:
        raise ValueError("Unknown mode", args.mode)


if __name__ == "__main__":
    main()
