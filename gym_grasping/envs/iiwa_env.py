"""
Environment class for running a (hardware) Kuka iiwa.
"""
import time
import math
from copy import deepcopy
import numpy as np
import cv2
import gym
from gym import spaces
from robot_io.kuka_iiwa.iiwa_controller import IIWAController
from robot_io.kuka_iiwa.wsg50_controller import WSG50Controller
from robot_io.cams.realsenseSR300_librs2 import RealsenseSR300


class FpsController:
    def __init__(self, freq):
        self.loop_time = 1.0 / freq
        self.prev_time = time.time()

    def step(self):
        current_time = time.time()
        delta_t = current_time - self.prev_time
        if delta_t < self.loop_time:
            time.sleep(self.loop_time - delta_t)
        self.prev_time = time.time()


class IIWAEnv(gym.Env):
    """
    Environment class for running a (hardware) Kuka iiwa.

    Args:
        dv: factor for cartesian position offset in relative cartesian position control, in meter, is multiplied
            with action, example: an action of [1,0,0,0,0] corresponds to a relative position change of dv meters in
            x direction
        drot: factor for orientation offset of gripper rotation in relative cartesian position control,
              in radians, is multiplied with 4th action component
              with action, example: an action of [1,0,0,0,0] corresponds to a relative position change of dv meters in
              x direction
        joint_vel:  max velocities of joint 1-6, range [0, 1], for PTP/joint motions
        gripper_rot_vel: max velocities of joint 7, , range [0, 1], for PTP/joint motions
        joint_acc: max acceleration of joint 1-7, range [0,1], for PTP/joint motions
        cartesian_vel: max translational and rotational velocity of EE, in mm/s, for LIN motions
        cartesian_acc: max translational and rotational acceleration of EE, in mm/s**2, for LIN motions
        freq: Control frequency in Hz
        act_type: policy actions are either ('multi-discrete', 'continuous')
        trajectory_type: Cartesian relative position control trajectory type ['ptp', 'lin']
        gripper_opening_threshold: discretization threshold for continuous control
        img_type: currently only RGB is supported
        obs_type: Image only, State only, Image & State ['image', 'state', 'image_state']
        use_impedance: use impedance control
        max_steps: max environment steps
        dof: degrees of freedom of cartesian control ['5dof', '7dof'], 5dof keeps Z-Axis vertical,
             only gripper roll rotation
        control: absolute or relative cartesian position control in TCP frame
        gripper_delay: delay for gripper actions
        safety_stop: Stop motion if force threshold is exceeded, move robot up
        safety_force_threshold: Force (N) and torque (Nm) threshold
        initial_gripper_state: ['open', 'close']
        reset_pose: Can be either 6-tuple cartesian pose or 7-tuple joint states (in degree) for reset
    """
    def __init__(self,
                 dv=0.01,
                 drot=0.2,
                 joint_vel=0.1,
                 gripper_rot_vel=0.3,
                 joint_acc=0.3,
                 cartesian_vel=100,
                 cartesian_acc=300,
                 freq=20,
                 act_type='continuous',  # ('multi-discrete', 'continuous')
                 trajectory_type='ptp',  # ['ptp', 'lin']
                 gripper_opening_threshold=0.6,  # TODO(lukas): Change to 0?
                 img_type='rgb',
                 obs_type='image',
                 use_impedance=False,
                 max_steps=100,
                 dof='5dof',
                 control='relative',  # 'relative', 'absolute'
                 gripper_delay=0,
                 safety_stop=True,
                 safety_force_threshold=20,  # Nm
                 initial_gripper_state='open',  # ['open', 'close']
                 reset_pose=(0, -0.56, 0.26, math.pi, 0, math.pi / 2)):  # (-90, 30, 0, -90, 0, 60, 0)

        self.discrete_control = False
        self.robot = IIWAController(use_impedance=use_impedance,
                                    joint_vel=joint_vel,
                                    gripper_rot_vel=gripper_rot_vel,
                                    joint_acc=joint_acc,
                                    cartesian_vel=cartesian_vel,
                                    cartesian_acc=cartesian_acc)
        self.gripper = WSG50Controller(max_opening_width=77)  # 109 for full
        self.cam = RealsenseSR300(img_type='rgb_depth')
        self.cam.set_rs_options(params={'white_balance': 3400.0,
                                        'exposure': 406.0, # 300
                                        'brightness': 60.0, # 50
                                        'contrast': 58.0, # 50
                                        'saturation': 64.0, # 64
                                        'sharpness': 50.0, # 50
                                        'gain': 45.0}) # 64
        assert trajectory_type in ['ptp', 'lin']
        assert not (dof == '7dof' and trajectory_type == 'lin')
        self.trajectory_type = trajectory_type
        assert dof in ['5dof', '7dof']
        self.dof = dof
        self.control = control
        self.navigate_in_cam_frame = True
        self.safety_stop = safety_stop
        self.safety_force_threshold = safety_force_threshold
        self.gripper_opening_threshold = gripper_opening_threshold
        self.keep_closed = False
        self.gripper_closing = False
        self.gripper_delay = gripper_delay
        assert initial_gripper_state in ['open', 'close']
        self.initial_gripper_state = 1 if initial_gripper_state == 'open' else -1
        self.gripper_queue = [self.initial_gripper_state] * self.gripper_delay
        self.dv = dv  # in mm
        self.drot = drot  # in rad
        self.freq = freq
        self.fps_controller = FpsController(freq)
        self._action_set = None
        self.set_up_action_space(act_type)
        # obs = self.reset()
        robot_state = np.zeros(12)
        self.arm_state = np.zeros(11)
        self.task_state = np.zeros(14)
        self.robot_info = None
        self.obs_type = obs_type
        assert len(reset_pose) in [6, 7]
        self.reset_pose = reset_pose
        if self.obs_type == 'image_state_reduced':
            robot_state = np.zeros(4)
            self.arm_state = self.arm_state[:6]
        self.resolution = (84, 84)
        self.observation = None
        if img_type == 'rgb':
            num_ch = 3
        elif img_type == 'depth':
            num_ch = 1
        elif img_type == 'rgbd':
            num_ch = 4
        else:
            raise ValueError('unknown rgb type')
        if self.obs_type == 'image':
            obs_s = spaces.Box(low=0, high=255,
                               shape=(*self.resolution, num_ch), dtype='uint8')
            self.observation_space = obs_s

        elif self.obs_type in ['image_state', 'image_state_reduced']:
            img = spaces.Box(low=0, high=255, shape=(*self.resolution, num_ch),
                             dtype='uint8')
            robot_state = spaces.Box(-np.inf, np.inf, shape=robot_state.shape,
                                     dtype='float32')
            task_state = spaces.Box(-np.inf, np.inf,
                                    shape=self.task_state.shape,
                                    dtype='float32')
            self.observation_space = spaces.Dict({'img': img,
                                                  'robot_state': robot_state,
                                                  'task_state': task_state})
        self._ep_step_counter = 0
        self.max_steps = max_steps
        self.camera_calibration = self.cam.get_intrinsics()

    def set_up_action_space(self, act_type):
        if act_type == 'multi-discrete':
            self.discrete_control = True

            class ActionConversion:
                """Mini-class for dict lookup API"""

                def __getitem__(self, discrete_action):
                    cont_action = np.array(discrete_action) - 1
                    # gripper action is only binary, robot actions have three components
                    assert discrete_action[-1] < 2
                    cont_action[-1] = -1 if discrete_action[-1] == 0 else 1
                    return cont_action
            self._action_set = ActionConversion()
            self.action_space = spaces.MultiDiscrete([3, ] * 4 + [2, ])
        elif act_type == 'continuous':
            action_dim = 5  # x y z + rot + gripper
            action_high = np.ones(action_dim)
            self.action_space = spaces.Box(-action_high, action_high,
                                           dtype=np.float32)

    def process_robot_state(self, state):
        if self.dof == '5dof':
            state = np.concatenate((state[0:4], state[6:13]))
        else:
            state = np.array(state)
        return state

    def process_action_robot(self, action):
        action = np.array(action)
        if self.control == 'relative':
            dxyz = action[0:3] * self.dv
            if self.navigate_in_cam_frame:
                gripper_orn = self.robot.get_tcp_pose()[5] + math.pi
                rot_mat = [[np.cos(gripper_orn), -np.sin(gripper_orn), 0],
                           [np.sin(gripper_orn), np.cos(gripper_orn), 0],
                           [0, 0, 1]]
                dxyz = np.matmul(np.array(rot_mat), dxyz)
            if self.dof == '5dof':
                coords = (*list(dxyz), 0, 0, action[3] * self.drot)
            else:
                coords = (*list(dxyz),
                          action[3] * self.drot,
                          action[4] * self.drot,
                          action[5] * self.drot)
            # state is defined in sendInfoMessage() in UDPIIWAJavaController
            # x,y,z,rot_x,rot_y,rot_z,joint[0-6], desired tcp pos x,y,z, rot_z, force x,y,z, torque x,y,z
            if self.trajectory_type == 'ptp':
                self.robot_info = self.robot.send_cartesian_coords_rel_PTP(coords)
            else:
                self.robot_info = self.robot.send_cartesian_coords_rel_LIN(coords)
        elif self.control == 'absolute':
            if np.all(action[:4] == 0):
                self.robot_info = self.robot.get_info()
            else:
                coords = (action[0], action[1], action[2], math.pi, 0, action[3])
                self.robot_info = self.robot.send_cartesian_coords_abs_PTP(coords)
                while not self.robot.reached_position(coords):
                    time.sleep(0.1)
                self.robot_info = self.robot.get_info()
        else:
            raise ValueError

        # if external force torque gets too high, move up and end program
        ext_force_exceeded = np.any(np.abs(self.robot_info['force_torque']) > self.safety_force_threshold)
        if self.safety_stop and ext_force_exceeded:
            if self.trajectory_type == 'ptp':
                self.robot.send_cartesian_coords_rel_PTP((0, 0, 0.06, 0, 0, 0))
            else:
                self.robot.send_cartesian_coords_rel_LIN((0, 0, 0.06, 0, 0, 0))
            print("WARNING: external force exceeded limit of 20 Nm")
            print(self.robot_info['force_torque'])
            exit()

    def process_action_gripper(self, action):
        gripper_action = action[-1]

        self.gripper_queue.append(gripper_action)
        gripper_action = self.gripper_queue.pop(0)
        if gripper_action > self.gripper_opening_threshold \
           and self.gripper_closing and not self.keep_closed:
            self.gripper.open_gripper()
            self.gripper_closing = False
        elif gripper_action < self.gripper_opening_threshold \
                and not self.gripper_closing:
            self.gripper.close_gripper()
            self.gripper_closing = True
        self.gripper.request_opening_width_and_force()

    def reset(self):
        self._ep_step_counter = 0
        self.gripper.open_gripper()
        self.gripper_closing = False
        self.gripper_queue = [self.initial_gripper_state] * self.gripper_delay
        self.gripper.request_opening_width_and_force()
        if len(self.reset_pose) == 6:
            #self.robot.send_cartesian_coords_rel_PTP([0, 0, 0.05, 0, 0, 0])
            #time.sleep(1.0)
            self.robot.send_cartesian_coords_abs_PTP(self.reset_pose)
            while not self.robot.reached_position(self.reset_pose):
                time.sleep(0.1)
        else:
            self.robot.send_joint_angles(self.reset_pose)
            while not self.robot.reached_joint_state(np.radians(self.reset_pose)):
                time.sleep(0.1)
        time.sleep(3)
        # self.gripper.close_gripper()
        # self.gripper_closing = True
        # time.sleep(3)
        self.robot_info = self.robot.get_info()
        return self._get_obs()[0]

    def reset_at_random_position(self, reset_joint_angles=False):
        self._ep_step_counter = 0
        # sample random position
        self.gripper.open_gripper()
        self.gripper_closing = False
        self.gripper_queue = [1] * self.gripper_delay
        self.gripper.request_opening_width_and_force()
        pos_x = np.random.uniform(-0.05, 0.05)
        pos_y = np.random.uniform(-0.6, -0.55)
        angle = np.random.uniform(-math.pi / 8, math.pi / 8)
        if reset_joint_angles:
            self.robot.send_joint_angles((-90, 40, 0, -70, 0, 70, 0))
        else:
            cart_cords = (pos_x, pos_y, 0.21, math.pi, 0,
                          math.pi/4 + math.pi / 6 + angle)
            self.robot.send_cartesian_coords_abs_PTP(cart_cords)
        time.sleep(5)
        self.robot_info = self.robot.get_info()
        return self._get_obs()[0]

    def render(self, mode='human'):
        if mode == 'human':
            if self.obs_type == 'image_state':
                img = cv2.resize(self.observation['img'], (500, 500))
            else:
                img = cv2.resize(self.observation, (500, 500))
            img = img[:, :, ::-1]
            cv2.imshow("win", img)
            cv2.waitKey(1)

    def _create_robot_observation(self):
        remaining_episode_length = (self.max_steps - self._ep_step_counter) / self.max_steps
        self.gripper_opening_width = self.gripper.get_opening_width()
        robot_state_full = np.array([*self.robot_info['tcp_pose'],
                                     *self.robot_info['joint_positions'],
                                     *self.robot_info['desired_tcp_pose'],
                                     *self.robot_info['force_torque'],
                                     self.gripper_opening_width,
                                     remaining_episode_length])
        robot_info = {'robot_state_full': robot_state_full, **self.robot_info,
                      'gripper_opening_width': self.gripper_opening_width,
                      'remaining_episode_length': remaining_episode_length}
        if self.obs_type == 'image_state':
            robot_state = np.array([*robot_state_full[0:3],
                                    *robot_state_full[5:13],
                                    *robot_state_full[17:]])
        elif self.obs_type == 'image_state_reduced':
            robot_state = robot_state_full[[2, 12, 23, 24]]
            rot_limits = [(-np.radians(175)) * 0.9, (np.radians(175)) * 0.9]
            # normalize
            robot_state[0] -= 0.13
            robot_state[0] /= 0.17
            robot_state[1] -= rot_limits[0]
            robot_state[1] /= (rot_limits[1] - rot_limits[0])
            robot_state[2] /= 0.07
        else:
            robot_state = robot_state_full.copy()
        return robot_state, robot_info

    def _get_obs(self):
        robot_state, robot_info = self._create_robot_observation()
        # do not crop unscaled image for max flow control
        rgb_unscaled, depth = self.cam.get_image(crop=False, flip_image=True)
        assert rgb_unscaled.shape == (480, 640, 3)
        rgb = rgb_unscaled[:, 80:560, :]
        rgb = cv2.resize(rgb, self.resolution)
        if self.obs_type == 'image':
            self.observation = rgb
        elif self.obs_type in ['image_state', 'image_state_reduced']:
            self.observation = {'img': rgb,
                                'robot_state': robot_state,
                                'task_state': self.task_state}
        info = {'rgb_unscaled': rgb_unscaled.copy(),
                'depth': depth.copy(),
                **deepcopy(robot_info)}
        return self.observation, info

    # @timeit
    def step(self, action):
        if not np.all(np.isfinite(action)):
            raise ValueError

        self._ep_step_counter += 1
        if not self.discrete_control and self.control == 'relative':
            action = np.clip(action, -1, 1)
        if self.discrete_control:
            action = self._action_set[action]
        self.process_action_gripper(action)
        self.process_action_robot(action)
        obs, info = self._get_obs()
        done, rew = self._check_done()
        # if done:
        #     self.keep_closed = True
        self.fps_controller.step()
        return obs, rew, done, info

    def _check_done(self):
        if self._ep_step_counter == 0:
            return False, 0
        done = False
        rew = 0
        if self._ep_step_counter > self.max_steps:
            done = True

        # force_threshold = 5
        # force = self.gripper.get_force()
        # if force > force_threshold and self.gripper_opening_width > 0.02:
        #     done = True
        #     rew = 1
        return done, rew
