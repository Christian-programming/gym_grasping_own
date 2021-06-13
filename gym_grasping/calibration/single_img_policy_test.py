"""
For a single real image, see what the policy output is.
"""
import cv2
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.ppo2.ppo2_eval_combi import get_ppo2_runner
from matplotlib.widgets import Slider

from gym_grasping.envs.robot_sim_env import RobotSimEnv
from robot_io.input_devices.space_mouse import SpaceMouse


class VisionPolicyTest:
    def __init__(self, snapshot_fn):
        env = RobotSimEnv(robot='kuka', task='block', initial_pose='close', act_type='continuous',
                          renderer='tiny',
                          act_dv=0.001, obs_type='image')
        env_v = DummyVecEnv([lambda: env])
        self.runner = get_ppo2_runner(env_v, nsteps=1, load_path=snapshot_fn)
        self.gripper_closing_threshold = 0.6

    def run_3dmouse(self):
        mouse = SpaceMouse()
        env = self.runner.env
        prev_gripper_ac = 1
        while 1:
            action = mouse.handle_mouse_events()
            mouse.clear_events()
            obs, _, _, _ = env.step([action])
            if action[4] == -1 and prev_gripper_ac == 1:
                self.plot(obs[0])
            prev_gripper_ac = action[4]

    def play_trajectory(self, filename):
        data = np.load(filename)
        # initial_configuration = data["arr_0"]
        # actions = data["arr_1"]
        # state_obs = data["arr_2"]
        img_obs = data["arr_3"]
        i = 0
        while 1:
            img = img_obs[i]
            resized = cv2.resize(img[:, :, ::-1], (500, 500))
            cv2.imshow("window", resized)
            k = cv2.waitKey(0)
            if (k % 256 == 97 or k % 256 == 65) and i >= 0:
                i -= 1
            elif (k % 256 == 100 or k % 256 == 68) and i < len(img_obs) - 1:
                i += 1
            elif k % 256 == 87 or k % 256 == 119:
                self.plot(img)

    def policy_act(self, obs):
        action, _, _, _ = self.runner.model.step([obs], None, False)
        action = np.clip(action[0], -1, 1)
        return action

    def visualize_action(self, action):
        h, w = 84, 84
        c_x, c_y = w // 2, h // 2

        def draw_vert_bar(img, start, end, c_x, c_y, width=2):
            img = img.copy()
            if end < start:
                start, end = end, start
            img[start: end, c_x - width // 2:c_x + width // 2] = 0
            return img

        def draw_hor_bar(img, start, end, c_x, c_y, width=2):
            img = img.copy()
            if end < start:
                start, end = end, start
            img[c_y - width // 2:c_y + width // 2, start:end] = 0
            return img

        x_y = np.ones((h, w, 3))
        x_y = draw_vert_bar(x_y, c_y, int(c_x - action[0] * h // 2), c_x, c_y)
        x_y = draw_hor_bar(x_y, int(c_y - action[1] * w // 2), c_x, c_x, c_y)

        z_rot = np.ones((h, w, 3))
        z_rot = draw_vert_bar(z_rot, c_y, int(c_x - action[2] * h // 2), c_x, c_y)
        z_rot = draw_hor_bar(z_rot, int(c_y + action[3] * w // 2), c_x, c_x, c_y)

        action[4] = (((np.clip(action[4], self.gripper_closing_threshold,
                               1) - self.gripper_closing_threshold) /
                      (1 - self.gripper_closing_threshold)) * 2) - 1
        op = np.ones((h, w, 3))
        op = draw_hor_bar(op, c_x - int((action[4] + 1) / 2 * (w / 2)),
                          c_x + int((action[4] + 1) / 2 * (w / 2)), c_x, c_y)

        return x_y, z_rot, op

    def plot(self, img):
        # rgb = rgb.resize((84, 84))
        original_img = img.copy()
        img = Image.fromarray(original_img)
        fig = plt.figure()
        ax1 = fig.add_subplot(241)
        ax2 = fig.add_subplot(242)
        ax3 = fig.add_subplot(243)
        ax4 = fig.add_subplot(244)
        fig.subplots_adjust(left=0, bottom=0)

        im1 = ax1.imshow(img)
        im2 = ax2.imshow(self.visualize_action(np.zeros(5))[0])
        im3 = ax3.imshow(self.visualize_action(np.zeros(5))[0])
        im4 = ax4.imshow(self.visualize_action(np.zeros(5))[0])

        bright = fig.add_axes([0.25, 0, 0.65, 0.03])
        con = fig.add_axes([0.25, 0.05, 0.65, 0.03])
        col = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        sharp = fig.add_axes([0.25, 0.15, 0.65, 0.03])
        zoom = fig.add_axes([0.25, 0.2, 0.65, 0.03])
        blur = fig.add_axes([0.25, 0.25, 0.65, 0.03])
        hue = fig.add_axes([0.25, 0.3, 0.65, 0.03])

        sbright = Slider(bright, 'bright', 0, 2, valinit=1)
        scon = Slider(con, 'con', 0, 2, valinit=1)
        scolor = Slider(col, 'color', 0, 2, valinit=1)
        ssharp = Slider(sharp, 'sharp', 0, 2, valinit=1)
        szoom = Slider(zoom, 'zoom', 0, 50, valinit=0)
        sblur = Slider(blur, 'blur', 0, 20, valinit=0)
        shue = Slider(hue, 'hue', -0.2, 0.2, valinit=0)

        def update(val):
            # rgb = original_img.copy()
            image = Image.fromarray(original_img)
            contrast = ImageEnhance.Contrast(image)
            image = contrast.enhance(scon.val)
            brightness = ImageEnhance.Brightness(image)
            image = brightness.enhance(sbright.val)
            color = ImageEnhance.Color(image)
            image = color.enhance(scolor.val)
            for _ in range(5):
                sharpness = ImageEnhance.Sharpness(image)
                image = sharpness.enhance(ssharp.val)
            blur_fac = int(sblur.val) * 2 + 1
            image = cv2.GaussianBlur(np.array(image), ksize=(blur_fac, blur_fac), sigmaX=0)
            image = image / 255
            img_hsv = matplotlib.colors.rgb_to_hsv(image)
            img_hsv[:, :, 0] += shue.val
            image = matplotlib.colors.hsv_to_rgb(img_hsv)
            if int(szoom.val) > 0:
                image = cv2.resize(image, (
                    image.shape[0] + int(szoom.val) * 2, image.shape[1] + int(szoom.val) * 2))
                image = image[int(szoom.val):-int(szoom.val), int(szoom.val):-int(szoom.val)]

            im1.set_array(image)
            obs = np.array(image * 255, dtype=np.uint8)
            action = self.policy_act(obs)
            print(action)

            x_y, z_rot, grip = self.visualize_action(action)
            im2.set_array(x_y)
            im3.set_array(z_rot)
            im4.set_array(grip)

            fig.canvas.draw()

        sbright.on_changed(update)
        scon.on_changed(update)
        scolor.on_changed(update)
        ssharp.on_changed(update)
        szoom.on_changed(update)
        sblur.on_changed(update)
        shue.on_changed(update)
        plt.show()


if __name__ == "__main__":
    VIS = VisionPolicyTest("/home/kuka/Downloads/02000(2)")
    # vis.run_3dmouse()
    VIS.play_trajectory(
        "/home/kuka/lang/robot/gym_grasping/gym_grasping/recordings/data/episode_3.npz")
