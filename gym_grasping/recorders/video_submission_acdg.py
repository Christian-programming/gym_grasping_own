import cv2
import pybullet as p
from a2c_ppo_acktr.play_model import Model, build_env, render_obs
from gym_grasping.envs import AdaptiveCurriculumEnv
import os
import numpy as np


def record():
    # save_path = '/media/kuka/Seagate Expansion Drive/kuka_recordings/kuka_icra_paper/' \
    #             'video_submission/box/shaped'
    # snapshot = "/mnt/home_hermannl/master_thesis/new_experiments/box_small/" \
    #            "shaped_no_contact_pen/" \
    #            "2019-09-11-19-02-38_110/save/ppo/box_small_no_curriculum_shaped-v0_2440.pt"
    # save_path = '/media/kuka/Seagate Expansion Drive/kuka_recordings/kuka_icra_paper/' \
    #             'video_submission/box/sparse'
    # snapshot = '/mnt/home_hermannl/master_thesis/new_experiments/box_small/no_curr_sparse/' \
    #            '2019-09-11-19-08-19_80/save/ppo/box_small_no_curriculum-v0_2440.pt'
    # save_path = '/media/kuka/Seagate Expansion Drive/kuka_recordings/kuka_icra_paper/' \
    #             'video_submission/box/bc_init'
    # snapshot = '/mnt/home_hermannl/master_thesis/new_experiments/bc_init/' \
    #            '2019-09-10-00-53-32_50/' \
    #            'save/ppo/box_small_no_curriculum-v0_2440.pt'
    # save_path = '/media/kuka/Seagate Expansion Drive/kuka_recordings/kuka_icra_paper/' \
    #             'video_submission/box/bc'
    # snapshot = '/mnt/home_hermannl/master_thesis/new_experiments/behavior_cloning/box/' \
    #            '2019-09-07-19-07-52/box_small-v0_200.pt'
    save_path = '/media/kuka/Seagate Expansion Drive/kuka_recordings/kuka_icra_paper/' \
                'video_submission/stacking/acgd'
    snapshot = '/mnt/home_hermannl_vision/raid/hermannl/new_experiments/stacking/10_09/' \
               'stackVel_primitive_05/2019-09-10-16-18-22_80/save/ppo/' \
               'stackVel_primitive_05-v0_2440.pt'

    env1 = AdaptiveCurriculumEnv(task='stackVel', curr='changing2', initial_pose='close',
                                 act_type='continuous', renderer='debug',
                                 obs_type='image_state_reduced', max_steps=150,
                                 restitution=0.5, gripper_delay=12,
                                 adaptive_task_difficulty=True, table_surface='white',
                                 position_error_obs=False, block_type='primitive', img_size="rl",
                                 movement_calib="new")
    # env1.env_params.set_variable_difficulty_mu('sim/restitution', 1)
    env1.seed(10)
    env = build_env(env1, normalize_obs=False)
    model = Model(env, snapshot, deterministic=True)

    obs = env.reset()
    done = False
    i = 0
    success = 0
    reward = 0
    ep_rews = 0
    j = 0
    os.mkdir(os.path.join(save_path, "episode_{}".format(i)))
    fourcc = cv2.VideoWriter_fourcc(*'jpeg')
    cv_vid_writer = cv2.VideoWriter(os.path.join(save_path, "vid_{}.mov".format(i)), fourcc, 25.0,
                                    (84, 84), True)
    while True:
        env1.eval = True
        action = model.step(obs, done)
        obs, rew, done, info = env.step(action)
        render_obs(obs, sleep=1)
        img = obs['img'].cpu().numpy()[0, ::-1, :, :].transpose((1, 2, 0)).astype(np.uint8)
        cv2.imwrite(os.path.join(save_path, "episode_{}/img_{}.jpg".format(i, j)), img)
        cv_vid_writer.write(img)
        done = done.any() if isinstance(done, np.ndarray) else done
        ep_rews += rew.cpu().flatten().numpy()[0]
        j += 1
        if done:
            if info[0]['task_success']:
                success += 1
            reward += ep_rews
            i += 1
            cv_vid_writer.release()
            cv_vid_writer = cv2.VideoWriter(os.path.join(save_path, "vid_{}.mov".format(i)), fourcc,
                                            25.0, (84, 84), True)
            os.mkdir(os.path.join(save_path, "episode_{}".format(i)))
            print("{} of {} successful, successrate: {}, avg reward: {} reward: {}, ep_rew: "
                  "{}, ep len: {}".format(success, i, success / i, reward / i, rew, ep_rews, j))
            j = 0
            obs = env.reset()
            ep_rews = 0


def replay_demonstration():
    save_path = '/media/kuka/Seagate Expansion Drive/kuka_recordings/kuka_icra_paper/' \
                'video_submission/stacking/bc_init'
    env = AdaptiveCurriculumEnv(robot='kuka', task='stackVel', renderer='egl',
                                act_type='continuous', initial_pose='close',
                                max_steps=150, obs_type='image_state', camera_pos='new_mount',
                                table_surface='white', block_type='primitive',
                                data_folder_path=os.path.join(save_path, 'tmp'))

    for i in range(1, 11):
        os.mkdir(os.path.join(save_path, "episode_{}".format(i)))
        fourcc = cv2.VideoWriter_fourcc(*'jpeg')
        cv_vid_writer = cv2.VideoWriter(os.path.join(save_path, "vid_{}.mov".format(i)), fourcc,
                                        25.0, (84, 84), True)
        env.reset()
        for j in range(150):
            env.reset_from_file(i, j)
            img = env.render()[:, :, ::-1]
            cv2.imshow("win", cv2.resize(img, (300, 300)))
            cv2.waitKey(1)
            cv2.imwrite(os.path.join(save_path, "episode_{}/img_{}.jpg".format(i, j)), img)
            cv_vid_writer.write(img)
        cv_vid_writer.release()


def record_green():
    save_path = '/media/kuka/Seagate Expansion Drive/kuka_recordings/kuka_icra_paper/' \
                'video_submission/stacking/bc_init'
    snapshot = '/mnt/home_hermannl/master_thesis/training_logs/2019-04-01-22-13-07/save/ppo/' \
               'changing2_diff_reg_egl_stackVel-v0_2440.pt'
    env1 = AdaptiveCurriculumEnv(task='stackVel', curr='changing2', initial_pose='close',
                                 act_type='continuous',
                                 renderer='egl',
                                 obs_type='image_state_reduced', max_steps=200,
                                 restitution=0.5, gripper_delay=12,
                                 adaptive_task_difficulty=True, table_surface='green',
                                 position_error_obs=False,
                                 block_type='primitive', img_size="rl", movement_calib="old")
    # env1.env_params.set_variable_difficulty_mu('sim/restitution', 1)
    env1.seed(10)
    env = build_env(env1, normalize_obs=False)
    model = Model(env, snapshot, deterministic=True)

    obs = env.reset()
    done = False
    i = 0
    success = 0
    reward = 0
    ep_rews = 0
    j = 0
    gripper_states = []
    ee_angles = []
    ee_positions = []
    try:
        os.mkdir(os.path.join(save_path, 'tmp/episode_{}'.format(i)))
    except FileExistsError:
        pass
    while True:
        env1.eval = True
        action = model.step(obs, done)
        obs, rew, done, info = env.step(action)
        render_obs(obs, sleep=1)
        env1._p.saveBullet(bulletFileName=os.path.join(save_path, "tmp/episode_{}/bullet_state_"
                                                                  "{}.bullet".format(i, j)))
        gripper_state, ee_angle, ee_pos, workspace = env1.get_simulator_state()
        gripper_states.append(gripper_state)
        ee_angles.append(ee_angle)
        ee_positions.append(ee_pos)
        done = done.any() if isinstance(done, np.ndarray) else done
        ep_rews += rew.cpu().flatten().numpy()[0]
        j += 1
        if done:
            if info[0]['task_success']:
                success += 1
            reward += ep_rews
            np.savez(
                os.path.join(save_path, "tmp/episode_{}/episode_{}.npz".format(i, i)),
                gripper_states=gripper_states,
                ee_angles=ee_angles,
                ee_positions=ee_positions,
                workspace=env1.robot.workspace)
            i += 1
            try:
                os.mkdir(os.path.join(save_path, 'tmp/episode_{}'.format(i)))
            except FileExistsError:
                pass
            print("{} of {} successful, successrate: {}, avg reward: {} reward: {}, ep_rew: {}, "
                  "ep len: {}".format(success, i, success / i, reward / i, rew, ep_rews, j))
            j = 0
            obs = env.reset()
            ep_rews = 0


def record_mouse():
    save_path = '/media/kuka/Seagate Expansion Drive/kuka_recordings/kuka_icra_paper/' \
                'video_submission/dr'

    from robot_io.input_devices.space_mouse import SpaceMouse
    env = AdaptiveCurriculumEnv(task='box_small', renderer='egl', act_type='continuous',
                                initial_pose='close', curr='changing2box',
                                max_steps=20000, obs_type='image_state_reduced', param_randomize=False,
                                camera_pos="new_mount", zoom=1,
                                position_error_obs=False, table_surface='white_box_bright',
                                block_type='model', restitution=0.5, img_size="168",
                                color_space='rgb', movement_calib="old", light_direction="new")
    mouse = SpaceMouse()
    d = 0
    j = 1
    record = False
    # env.params.set_variable_difficulty_mu("sim/restitution", 0)
    env.curr.set_difficulty(env.params, 0, False)
    gripper_state, ee_angle, ee_pos, workspace = None, None, None, None
    fourcc = cv2.VideoWriter_fourcc(*'jpeg')

    cv_vid_writer = None  # will be set in loop
    while 1:
        ret = 0
        for i in range(150000):
            action = mouse.handle_mouse_events()
            mouse.clear_events()
            ob, reward, done, info = env.step(action)
            # print(reward)
            ret += reward
            # print(ob['robot_state'][0])
            img = ob['img'][:, :, ::-1]
            if record:
                cv_vid_writer.write(img)
            img = cv2.resize(img, (300, 300))
            cv2.imshow("win", img)
            k = cv2.waitKey(10) % 256
            if k == ord('r'):
                record = True
                cv_vid_writer = cv2.VideoWriter(os.path.join(save_path, "vid_{}.mov".format(j)),
                                                fourcc, 25.0, (84, 84), True)
            if k == ord('e'):
                record = False
                cv_vid_writer.release()
                j += 1
            if k == ord('a'):
                d += 0.1
                d = np.clip(d, 0, 1)
                # env.params.set_variable_difficulty_mu("sim/restitution", d)
                env.curr.set_difficulty(env.params, d, False)
                print(d)
            if k == ord('q'):
                d -= 0.1
                d = np.clip(d, 0, 1)
                # env.params.set_variable_difficulty_mu("sim/restitution", d)
                env.curr.set_difficulty(env.params, d, False)
                print(d)
            if k == ord('d'):
                # denv.reset()
                # print(env.params.restitution())
                break
            if k == ord('s'):
                p.saveBullet(bulletFileName=os.path.join(save_path, "tmp.bullet"))
                gripper_state, ee_angle, ee_pos, workspace = env.get_simulator_state()
            # env.render()
            if k == ord('l'):
                p.restoreState(fileName=os.path.join(save_path, "tmp.bullet"))
                env.robot.gripper.reset(gripper_open=gripper_state)
                env.robot.desired_ee_angle = ee_angle
                env.robot.desired_ee_pos = ee_pos
                env.robot.workspace = workspace
                env._task.reset_from_curriculum()
            # if done:
            #     # print(ret)
            #     print("steps: {}".format(i))
            #     env.reset()
            #     break
        env.reset()


def record_curriculum():
    from robot_io.input_devices.space_mouse import SpaceMouse
    from gym_grasping.envs import AdaptiveCurriculumEnv
    env = AdaptiveCurriculumEnv(task='stackVel', cur='changing2', renderer='debug',
                                obs_type='image_state',
                                max_steps=100,
                                use_regular_starts=False, adaptive_task_difficulty=True,
                                reg_start_func='f_reg_prog',
                                use_diff_reg=True, table_surface='white',
                                data_folder_path='/home/kuka/lang/robot/gym_grasping/gym_grasping/'
                                                 'curriculum/data_stack_slow')
    mouse = SpaceMouse()
    d = 0
    a = 0
    j = 0
    save_path = '/media/kuka/Seagate Expansion Drive/kuka_recordings/kuka_icra_paper/' \
                'video_submission/curriculum'
    data = {'update_step': 5,
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
    record_vid = False
    fourcc = cv2.VideoWriter_fourcc(*'jpeg')
    cv_vid_writer = None  # will be set in loop
    while 1:
        for i in range(100):
            action = mouse.handle_mouse_events()
            mouse.clear_events()
            ob, reward, done, info = env.step(action)
            # print(ob['robot_state'][-1])
            img = ob['img'][:, :, ::-1]
            cv2.imshow("win", cv2.resize(img, (300, 300)))
            k = cv2.waitKey(30) % 256
            if record_vid:
                cv_vid_writer.write(img)
            if k == ord('r'):
                record_vid = True
                cv_vid_writer = cv2.VideoWriter(os.path.join(save_path, "vid_{}.mov".format(j)),
                                                fourcc, 25.0, (84, 84), True)
            if k == ord('e'):
                record_vid = False
                cv_vid_writer.release()
                j += 1
            if k == ord('d'):
                d += 0.1
                print(d)
                data['difficulty_cur'] = d
            if k == ord('a'):
                a += 0.1
                print(a)
                data['difficulty_reg'] = a
            # if k == ord('r'):
            #     info = env.reset(data)
            #     # info = env.params.get_curriculum_info()
            #     print(env._info)
            #     # for k,v in info.items(): print(k,v)
            #     print()
            #     # env.render()
            if done:
                env.reset(data)


if __name__ == "__main__":
    # record()
    # record_mouse()
    # record_green()
    # replay_demonstration()
    record_curriculum()
