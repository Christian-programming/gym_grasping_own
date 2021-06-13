import time
import math
import cv2
from gym_grasping.envs.iiwa_env import IIWAEnv
from gym_grasping.recorders.dataset import DatasetRecorder
from robot_io.input_devices.space_mouse import SpaceMouse


def main():
    mode = 'enjoy_pytorch'

    if mode == "enjoy_pytorch":
        from a2c_ppo_acktr.play_model import Model, build_env, render_obs

        iiwa = IIWAEnv(act_type='multi-discrete', freq=20, obs_type='image_state_reduced',
                       dv=0.005, drot=0.1, use_impedance=True,
                       max_steps=250, gripper_delay=3,  reset_pose=((-90, 30, 0, -85, 0, 65 , 0)))
        # snapshot = "/tmp/home/dobrusii/Masterthesis/Thesis/experiments/011_train_on_consistency_only/train/brown_v4/target_network_update/color_jitter_ablation/color_jitter_only/tau_1.0/2020-10-15_01-54_seed-80/save/consistency/stackVel_brown_v4_no_dr-v0_389.pt"
        # snapshot = "/tmp/home/dobrusii/Masterthesis/Thesis/experiments/011_train_on_consistency_only/train/brown_v4/target_network_update/color_jitter_ablation/randomized_magnitude_center_crop_color_jitter/tau_1.0/2020-10-14_18-24_seed-80/save/consistency/stackVel_brown_v4_no_dr-v0_389.pt"
        # snapshot = "/tmp/home/dobrusii/Masterthesis/Thesis/experiments/004_train_and_eval_aug_loss_from_dataset/train/baseline_cork_bg/with_dr/2020-07-12_19-10_seed-100/save/ppo/stackVel_cork_with_dr-v0_2440.pt"
        # snapshot = "/tmp/home/dobrusii/Masterthesis/Thesis/experiments/004_train_and_eval_aug_loss_from_dataset/train/baseline_brown_bg/with_dr/2020-07-12_19-46_seed-100/save/ppo/stackVel_brown_with_dr-v0_2440.pt"
        # snapshot = "/tmp/home/dobrusii/Masterthesis/Thesis/experiments/004_train_and_eval_aug_loss_from_dataset/train/baseline_cork_bg/with_dr/2020-07-12_19-10_seed-100/save/ppo/stackVel_cork_with_dr-v0_2440.pt"
        # snapshot = "/tmp/home/dobrusii/Masterthesis/Thesis/experiments/004_train_and_eval_aug_loss_from_dataset/train/baseline_cork_bg/no_dr/2020-07-10_20-46_seed-80/save/ppo/stackVel_cork_no_dr-v0_2440.pt"
        # snapshot = "/tmp/home/dobrusii/Masterthesis/Thesis/experiments/011_train_on_consistency_only/train/brown_v4/target_network_update/color_jitter_ablation/randomized_magnitude_aug_list_ours/tau_0.99/2020-10-15_09-53_seed-100/save/consistency/stackVel_brown_v4_no_dr-v0_389.pt"

        # snapshot = "/tmp/home/dobrusii/Masterthesis/Thesis/experiments/013_train_with_real_data/train/real/randomized_magnitude_center_crop_color_jitter/2020-10-22_23-10_seed-80/save/consistency/stackVel_cork_no_dr-v0_291.pt"
        snapshot = ('/home/kuka/lang/robot/gym_grasping/gym_grasping/scripts/policy/1453_2440.pt')

        # Wrap GraspingEnv in DictVecPytorchEnv first, because observations will be casted to torch tensors
        # This makes saving to tensorboard easier -> could be omitted...
        # data_dir = "/tmp/home/dobrusii/Masterthesis/Thesis/tmp/data"
        # img_dir = "/tmp/home/dobrusii/Masterthesis/Thesis/tmp/img"

        # iiwa = DatasetRecorder(env=iiwa, save_dir=data_dir, img_dir=img_dir,
        #                       store_individual_episode_steps=True,
        #                       save_pybullet_state=False)
        env = build_env(iiwa, normalize_obs=False)
        model = Model(env, snapshot, deterministic=True)
        # env = iiwa
        obs = env.reset()
        done = False
        while True:
            for i in range(200):
                action = model.step(obs, done)
                obs, rew, done, info = env.step(action)
                print(action.cpu().numpy())
                render_obs(obs)
                # rgb = kinect.get_image()
                # rgb = cv2.resize(rgb, (640, 360))
                # cv2.imshow("win2", rgb[:, 140:500,::-1])
                cv2.waitKey(1)
            obs = env.reset()

    elif mode == "3dmouse":
        iiwa = IIWAEnv(act_type='continuous', freq=20, obs_type='image_state_reduced', dv=0.01,
                       drot=0.04, joint_vel=0.05, trajectory_type='lin',
                       gripper_rot_vel=0.3, joint_acc=0.3, use_impedance=True, safety_stop=True,
                       dof='5dof',
                       #reset_pose=(0, -0.56, 0.26, math.pi, 0, math.pi / 2)
                       reset_pose=((-90, 30, 0, -85, 0, 65 , 0)))

        # data_dir = "/home/kuka/TmpDrive/cork_recordings/data"
        # img_dir = "/home/kuka/TmpDrive/cork_recordings/img"
        # iiwa = DatasetRecorder(env=iiwa, save_dir=data_dir, img_dir=img_dir,
        #                       store_individual_episode_steps=True,
        #                       save_pybullet_state=False)

        iiwa.reset()

        mouse = SpaceMouse(act_type='continuous', mode='5dof')

        while 1:
            action = mouse.handle_mouse_events()
            print('action', action)
            mouse.clear_events()
            ob, _, done, info = iiwa.step(action)
            # print(ob['robot_state'][2])
            # print(info['robot_state_full'][2])
            # cv2.imshow("win", cv2.resize(ob[:, :, ::-1], (500, 500)))
            img = info['rgb_unscaled'][:, :, ::-1]
            cv2.imshow('win', img)
            # cv2.waitKey(1)

            key = cv2.waitKey(10) % 256
            # if key == ord('s'):
            #     print("Resetting...")
            #     iiwa.reset()
            # if key == ord('r'):
            #     print("Resetting...without saving")
            #     iiwa.reset(skip_saving=True)


    elif mode == "absolute":
        iiwa = IIWAEnv(act_type='continuous', freq=20, obs_type='image_state_reduced', dv=0.01,
                       drot=0.2, joint_vel=0.1,
                       gripper_rot_vel=0.3, joint_acc=0.3, use_impedance=True,
                       dof='5dof', control='absolute')
        iiwa.reset()
        actions = [(0, -0.56, 0.3, math.pi / 2, 1), (0.05, -0.56, 0.32, math.pi / 2, 1),
                   (0, -0.59, 0.3, math.pi / 2, 1), (-0.03, -0.56, 0.31, math.pi / 2, 1)]

        for i in [i for i in range(4)] * 100:
            action = actions[i]
            ob, _, done, info = iiwa.step(action)
            # print(ob['robot_state'][2])
            print(info['robot_state_full'][17:23])
            # cv2.imshow("win", cv2.resize(ob['img'][:,:,::-1], (500,500)))
            img = info['rgb_unscaled'][:, :, ::-1]

            cv2.imshow('win', img)
            cv2.waitKey(1)
            time.sleep(2)


if __name__ == "__main__":
    main()
