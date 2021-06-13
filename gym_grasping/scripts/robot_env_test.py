import numpy as np
import cv2

cv2.imshow("win", np.zeros((300, 300)))
cv2.waitKey(1)
import pybullet as p
import matplotlib.pyplot as plt
from gym_grasping.envs.robot_sim_env import RobotSimEnv
import gym

def main():
    #
    # Chose what you want to do here!
    #
    mode = 'keyboard'

    if mode == "play":
        env = RobotSimEnv(robot='kuka', task='block_shaped', renderer='debug',
                          act_type='continuous', initial_pose='close',
                          max_steps=None, act_dv=0.01, obs_type='state')
        dv = 0.01
        motors_ids = [p.addUserDebugParameter("posX", -dv, dv, 0),
                      p.addUserDebugParameter("posY", -dv, dv, 0),
                      p.addUserDebugParameter("posZ", -dv, dv, 0),
                      p.addUserDebugParameter("roll", -dv, dv, 0),
                      p.addUserDebugParameter("grip", -1, 1, 1)]
        ret = 0
        done = False
        while 1:
            action = []
            for motorId in motors_ids:
                action.append(p.readUserDebugParameter(motorId))
            # round ac
            action = list(np.round(action, decimals=3))
            # uncomment if multi-discrete
            # mdisc_a = [0] * len(ac)
            # for i,a in enumerate(ac):
            #     if a < 0:
            #         mdisc_a[i] = 0
            #     elif a == 0:
            #         mdisc_a[i] = 1
            #     else:
            #         mdisc_a[i] = 2
            # ac = mdisc_a
            # env.reset()
            ob, reward, done, info = env.step(action)
            # print(i)
            env.render()

            if done:
                env.reset()
            ret += reward
        print(ret)

    elif mode == "3dmouse":
        from robot_io.input_devices.space_mouse import SpaceMouse
        env = RobotSimEnv(task='grasp', renderer='debug', act_type='continuous', control='relative',
                          initial_pose='close', max_steps=1500, obs_type='image_state_reduced', param_randomize=False,
                          camera_pos="new_mount", zoom=1, sample_params=True, table_surface='cork')
        keyboard = SpaceMouse(act_type='continuous')
        d = 0
        while 1:
            for i in range(1000):
                action = keyboard.handle_mouse_events()
                keyboard.clear_events()
                ob, reward, done, info = env.step(action)
                # print(ob['robot_state'])
                img = cv2.resize(ob['img'][:, :, ::-1], (300, 300))
                cv2.imshow("win", img)
                k = cv2.waitKey(10) % 256
                # k = 0
                # if k == ord('a'):
                #     env.params.set_variable_difficulty_r("geom/block_1", d)
                #     env.params.set_variable_difficulty_mu("object_to_gripper", tag="geom", d)
                #     d += 0.1
                # if k == ord('d'):
                #     env.reset()
                #     print(env.params.restitution())
                # env.render()
                if k == ord('r'):
                    env.reset()
                if done:
                    print(reward)
                    env.reset()
            env.reset()
    elif mode == "keyboard":
        from robot_io.input_devices.keyboard_input import KeyboardInput
        env = RobotSimEnv(task='stack_shaped', renderer='egl', act_type='continuous', control='relative',
                          initial_pose='close', max_steps=1500, obs_type='image', param_randomize=False,
                          camera_pos="new_mount")
        print("keyboard")
        keyboard = KeyboardInput()
        env_name = "kuka_block_grasping-v0"
        env  = gym.make(env_name, renderer='egl')
        d = 0
        env.reset()
        while 1:
            for i in range(1000):
                #action = keyboard.handle_keyboard_events()
                action = env.action_space.sample()
                print("action ", action)
                ob, reward, done, info = env.step(action)
                # print(ob['robot_state'])
                print(ob)
                #img = cv2.resize(ob['img'][:, :, ::-1], (300, 300))
                #img = cv2.resize(ob[:, :, ::-1], (300, 300))
                img = cv2.resize(ob, (300, 300)) / 255
                print(img)
                cv2.imshow("win", img)
                cv2.waitKey(0)
                k = 0
                # if k == ord('a'):
                #     env.params.set_variable_difficulty_r("geom/block_1", d)
                #     env.params.set_variable_difficulty_mu("object_to_gripper", tag="geom", d)
                #     d += 0.1
                # if k == ord('d'):
                #     env.reset()
                #     print(env.params.restitution())
                # env.render()
                if k == ord('r'):
                    env.reset()
                if done:
                    print(i)
                    print(reward)
                    env.reset()
            env.reset()

    elif mode == "absolute":
        from robot_io.input_devices.space_mouse import SpaceMouse
        env = RobotSimEnv(task='flow_stack', renderer='debug', act_type='continuous',
                          initial_pose='close', max_steps=500, obs_type='image_state', param_randomize=False,
                          camera_pos="new_mount", zoom=1, sample_params=False,
                          img_size=(256, 256), control='absolute')
        d = 0
        actions = [[0.62499413, -0.55500522, 0.5, 0.78539153, 1],
                   [0.52499475, -0.5049671, 0.5, 0.78539075, 1],
                   [0.42930337, - 0.69993019, 0.5, 0.12449774, 1]]
        for i in range(3):
            for _ in range(100):
                ob, reward, done, info = env.step(actions[i])
                # print(ob['robot_state'])
                img = cv2.resize(ob['img'][:, :, ::-1], (300, 300))
                cv2.imshow("win", img)
                k = cv2.waitKey(10) % 256
                # if k == ord('a'):
                #     env.params.set_variable_difficulty_r("geom/block_1", d)
                #     env.params.set_variable_difficulty_mu("object_to_gripper", tag="geom", d)
                #     d += 0.1
                # if k == ord('d'):
                #     env.reset()
                #     print(env.params.restitution())
                # env.render()
                if k == ord('r'):
                    env.reset()
                if done:
                    print(reward)
                    env.reset()

    elif mode == "curriculum":
        from robot_io.input_devices.space_mouse import SpaceMouse
        from gym_grasping.envs import AdaptiveCurriculumEnv
        env = AdaptiveCurriculumEnv(task='stackVel', curr='stack', renderer='debug',
                                    obs_type='image_state_reduced',
                                    max_steps=150,
                                    use_regular_starts=True, adaptive_task_difficulty=True,
                                    reg_start_func='f_sr_prog',
                                    data_folder_path='/home/kuka/lang/robot/gym_grasping/gym_grasping/curriculum/data_new')
        keyboard = SpaceMouse()
        d = 0
        a = 0

        data = {'update_step': 0,
                'num_updates': 10,
                'eprewmean': 0.5,
                'curr_eprewmean': 0,
                'eval_eprewmean': 0,
                'reg_eprewmean': 0,
                'eval_reg_eprewmean': 0,
                'difficulty_cur': d,
                'difficulty_reg': a,
                'reg_success_rate': 0,
                'curr_success_rate': 0}
        env.reset(data)
        while 1:
            for i in range(20):
                action = keyboard.handle_mouse_events()
                keyboard.clear_events()
                ob, reward, done, info = env.step(action)
                # print(ob['robot_state'][-1])
                img = cv2.resize(ob['img'][:, :, ::-1], (300, 300))
                cv2.imshow("win", img)
                k = cv2.waitKey(10) % 256
                # k = 0
                if k == ord('d'):
                    d += 0.1
                    print(d)
                    data['difficulty_cur'] = d
                if k == ord('a'):
                    a += 0.1
                    print(a)
                    data['difficulty_reg'] = a
                if k == ord('r'):
                    info = env.reset(data)
                    # info = env.params.get_curriculum_info()
                    # print(env._info)
                    # for k,v in info.items(): print(k,v)
                    # print(info['episode_info']['dyn/robot_dv'])
                    print(env.params.restitution)
                    # env.render()
                if done:
                    env.reset(data)
                    # print(info['episode_info']['sim/restitution'])
                    print(info['episode_info']['reset_type'])

    elif mode == "test_params":
        env = RobotSimEnv(task='stackVel', renderer='debug', act_type='continuous',
                          initial_pose='close', max_steps=None, obs_type='image_state', param_randomize=False,
                          camera_pos="new_mount", zoom=1)
        r = []
        for i in range(300):
            env.reset()
            r.append(env.params.restitution())
        plt.hist(r, bins=20)
        plt.show()

    elif mode == "test_performance":
        from a2c_ppo_acktr.play_model import Model, build_env, render_obs
        from gym_grasping.envs.curriculum_env import AdaptiveCurriculumEnv
        env = AdaptiveCurriculumEnv(task='stackVel', curr='stack', initial_pose='close',
                                    act_type='multi-discrete', renderer='egl',
                                    obs_type='image_state_reduced', max_steps=150,
                                    table_surface='white')
        # snapshot = ('/run/user/9984/gvfs/sftp:host=hpcgpu2,user=hermannl/home/hermannl/logs/gym_grasping/multi-discrete/fast/2020-09-10-20-23-45_100/save/ppo/stackVel_acgd_md_fast-v0_2440.pt')
        snapshot = ('/home/kuka/lang/robot/gym_grasping/gym_grasping/scripts/policy/1453_2440.pt')
        env.seed(10)
        env = build_env(env, normalize_obs=False)
        model = Model(env, snapshot, deterministic=True)

        obs = env.reset()
        done = False
        i = 0
        success = 0
        reward = 0
        ep_rews = 0
        while True:
            action = model.step(obs, done)
            obs, rew, done, info = env.step(action)

            render_obs(obs, sleep=1)
            done = done.any() if isinstance(done, np.ndarray) else done
            ep_rews += rew.cpu().flatten().numpy()[0]
            if done:
                if info[0]['task_success']:
                    success += 1
                reward += ep_rews
                i += 1
                print("{} of {} successful, successrate: {}, avg reward: {} reward: {}, "
                      "ep_rew: {}".format(success, i, success / i, reward / i, rew, ep_rews))
                obs = env.reset()
                ep_rews = 0

    else:
        raise ValueError("Unknown mode", mode)


if __name__ == "__main__":
    main()
