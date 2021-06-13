"""
record and replay trajectories on real robot.
"""
import csv
import os
import time

import cv2
import numpy as np
from PIL import Image

from gym_grasping.envs.iiwa_env import IIWAEnv
from robot_io.kuka_iiwa.iiwa_controller import IIWAController
from robot_io.cams.realsenseRS300_librs2 import RealsenseSR300
from robot_io.input_devices.space_mouse import SpaceMouse

# TODO(lukas): is this still relevant
# for using 7dof dof comment lines 252 & 253 in udpiiwajavacontroller


def record_trajectory():
    '''record a trajectory'''
    iiwa = IIWAEnv(act_type='continuous', freq=20, obs_type='image_state', dv=0.01, drot=0.1,
                   joint_vel=0.1,
                   gripper_rot_vel=0.3, joint_acc=0.3, use_impedance=False,
                   dof='7dof')
    iiwa.reset()
    mouse = SpaceMouse(act_type='continuous', mode='7dof')
    # from robot_io.cams.kinect2 import Kinect2
    # kinect = Kinect2()
    while 1:
        action = mouse.handle_mouse_events()
        mouse.clear_events()
        obs, _, _, _ = iiwa.step(action)
        cv2.imshow("win", cv2.resize(obs['img'][:, :, ::-1], (500, 500)))
        k = cv2.waitKey(1) % 256
        if k == ord('d'):
            print(obs['robot_state'])
            with open("../recordings/reconstruction/insertion/trajectory.csv", "a") as f_obj:
                csv_writer = csv.writer(f_obj)
                csv_writer.writerow(list(obs['robot_state']))


def replay_trajectory(record_path=None):
    '''replay a trajectory'''
    if not os.path.isdir(record_path):
        os.mkdir(record_path)

    joint_positions = []
    with open("../recordings/reconstruction/insertion/trajectory.csv", 'r') as f_obj:
        csv_reader = csv.reader(f_obj)
        for row in csv_reader:
            joint_positions.append(row[6:13])

    if record_path:
        cam = RealsenseSR300(img_type='rgb_depth')
    else:
        cam = None

    iiwa = IIWAController(use_impedance=False, joint_vel=0.1, gripper_rot_vel=0.3, joint_acc=0.3)
    iiwa.send_joint_angles(tuple(joint_positions[0]), mode="rad")
    time.sleep(3)
    for i, joint_pos in enumerate(joint_positions[1:]):
        iiwa.send_joint_angles(tuple(joint_pos), mode="rad")
        # TODO(max): set back to 1 for small_object, 5 for workspace
        time.sleep(5)
        state = iiwa.get_joint_info()
        with open(os.path.join(record_path, "robot_states.csv"), "a") as f_obj:
            csv_writer = csv.writer(f_obj)
            csv_writer.writerow(list(state))
        if cam:
            img, depth = cam.get_image(crop=False)
            pil_img = Image.fromarray(img)
            filename = "{0:04d}.jpg".format(i)
            path = os.path.join(record_path, filename)
            pil_img.save(path)
            np.savez(path.replace(".jpg", ""), depth)


if __name__ == '__main__':
    # record_trajectory()
    replay_trajectory("../recordings/reconstruction/insertion_nomarker/")
