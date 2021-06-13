"""
Compare simulation to images or real camera
"""
import cv2

from gym_grasping.envs.robot_sim_env import RobotSimEnv
from robot_io.cams.realsenseSR300_librs2 import RealsenseSR300
from robot_io.input_devices.space_mouse import SpaceMouse


def nothing(variable):
    '''do nothing'''
    pass
#



def main():
    env = RobotSimEnv(robot='kuka', task='stackVel', initial_pose='close', act_type='continuous',
                      renderer='egl', act_dv=0.001, obs_type='image', param_randomize=False,
                      table_surface='white', color_space='rgb', img_size="rl")
    cam = RealsenseSR300()
    mouse = SpaceMouse()

    # (.0, .67, .76), ul = (.01, .73, .85), f = hsv2rgb)
    # self.params.add_variable("block_blue", tag="vis", ll=(.55, .47, .35), ul=(.59, .51, .4), f=hsv2rgb)
    # self.params.add_variable("table_green", tag="vis", ll=(.2, .41, .66), ul=(.25, .47, .7),
    cv2.namedWindow("win")
    # red_hsv_id = [p.addUserDebugParameter("red_h", -1, 1, .011),
    #            p.addUserDebugParameter("red_s", 0, 1, .526),
    #            p.addUserDebugParameter("red_v", 0, 1, .721)]
    # blue_hsv_id = [p.addUserDebugParameter("blue_h", 0, 1, 0.57),
    #            p.addUserDebugParameter("blue_s", 0, 1, 0.49),
    #            p.addUserDebugParameter("blue_v", 0, 1, 0.375)]
    # table_hsv_id = [p.addUserDebugParameter("table_h", 0, 1, 0.15),
    #                p.addUserDebugParameter("table_s", 0, 1, 0.1),
    #                p.addUserDebugParameter("table_v", 0, 1, 0.95)]
    cv2.createTrackbar("red_h", "win", 11, 1000, nothing)
    cv2.createTrackbar("red_s", "win", 526, 1000, nothing)
    cv2.createTrackbar("red_v", "win", 721, 1000, nothing)
    cv2.createTrackbar("blue_h", "win", 550, 1000, nothing)
    cv2.createTrackbar("blue_s", "win", 236, 1000, nothing)
    cv2.createTrackbar("blue_v", "win", 405, 1000, nothing)
    cv2.createTrackbar("table_h", "win", 150, 1000, nothing)
    cv2.createTrackbar("table_s", "win", 100, 1000, nothing)
    cv2.createTrackbar("table_v", "win", 950, 1000, nothing)

    cv2.createTrackbar("exposure", "win", 300, 1000, nothing)
    cv2.createTrackbar("brightness", "win", 50, 100, nothing)
    cv2.createTrackbar("contrast", "win", 50, 100, nothing)
    cv2.createTrackbar("saturation", "win", 64, 100, nothing)
    cv2.createTrackbar("gain", "win", 64, 200, nothing)

    #rs_params={'white_balance': 3400.0,
                # 'exposure': 406.0,  # 300
                # 'brightness': 55.0,  # 50
                # 'contrast': 55.0,  # 50
                # 'saturation': 64.0,  # 64
                # 'sharpness': 50.0,  # 50
                # 'gain': 45.0})  # 64

    while 1:
        for _ in range(50000):
            # red_hsv = np.array(
            #    [cv2.getTrackbarPos("red_h", "win"), cv2.getTrackbarPos("red_s", "win"),
            #     cv2.getTrackbarPos("red_v", "win")]) / 1000
            # blue_hsv = np.array(
            #    [cv2.getTrackbarPos("blue_h", "win"), cv2.getTrackbarPos("blue_s", "win"),
            #     cv2.getTrackbarPos("blue_v", "win")]) / 1000
            # table_hsv = np.array(
            #    [cv2.getTrackbarPos("table_h", "win"), cv2.getTrackbarPos("table_s", "win"),
            #     cv2.getTrackbarPos("table_v", "win")]) / 1000

            # red = colorsys.hsv_to_rgb(*red_hsv) + (1,)
            # blue = colorsys.hsv_to_rgb(*blue_hsv) + (1,)
            # table = colorsys.hsv_to_rgb(*table_hsv) + (1,)
            # env.params.block_red = lambda : red
            # env.params.block_blue = lambda : blue
            # env.params.table_green = lambda : table

            e = cv2.getTrackbarPos("exposure", "win")
            b = cv2.getTrackbarPos("brightness", "win")
            c = cv2.getTrackbarPos("contrast", "win")
            s = cv2.getTrackbarPos("saturation", "win")
            g = cv2.getTrackbarPos("gain", "win")

            cam.set_rs_options(params={'white_balance': 3400.0,
                                        'exposure': e,
                                        'brightness': b,
                                        'contrast': c,
                                        'saturation': s,
                                        'sharpness': 50.0,
                                        'gain': g})

            action = mouse.handle_mouse_events()
            mouse.clear_events()
            obs, _, _, _ = env.step(action)
            cv2.imshow("win", cv2.resize(obs[:, :, ::-1], (480, 480)))
            img, _ = cam.get_image(flip_image=True, crop=True)
            # img = np.array(matplotlib.colors.rgb_to_hsv(np.array(img, dtype=np.float64) /
            # 255)*255, dtype=np.uint8)
            cv2.imshow("img", cv2.resize(cv2.resize(img[:, :, ::-1], (84, 84)), (480, 480)))
            k = cv2.waitKey(1) % 256
            if k == ord('d'):
                break
        env.reset()


if __name__ == "__main__":
    main()
