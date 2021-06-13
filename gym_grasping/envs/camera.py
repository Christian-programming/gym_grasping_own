'''Camera class to handle rendering'''
import math
from math import tan, pi

import numpy as np
import matplotlib.colors
from PIL import Image, ImageEnhance, ImageFilter


class PyBulletCamera:
    '''Camera class to handle rendering'''
    def __init__(self, p, env_params, img_size, img_type, robot_cam=True,
                 calibration=None):
        self.cid = None
        self.p = p
        self.params = env_params
        self.img_type = img_type
        assert calibration is not None

        self.camera_calibration = None
        self._camera_near = 0.01
        self._camera_far = 100.0

        self.robot_cam = robot_cam
        # set in set_robot
        self.robot_uid = None
        self.camera_index = None

        # set in reset
        self._base_view_matrix = None
        self._base_proj_matrix = None
        self.zoom = None

        # init for get_buffer_size
        if img_size == 'rl':
            self.width, self.height = 84, 84
        elif img_size == 'video':
            self.width, self.height = 480, 360
        elif isinstance(img_size, tuple):
            assert len(img_size) == 2
            self.width, self.height = img_size
        else:
            raise ValueError
        self.img_size = self.width, self.height

        # config used to be in Env
        self.params.add_variable("zoom", 1, tag="cam")
        uos = False
        self.params.add_variable("brightness", tag="vis", ll=.9, ul=1.1, update_on_step=uos)
        self.params.add_variable("contrast", tag="vis", ll=.9, ul=1.1, update_on_step=uos)
        self.params.add_variable("color", tag="vis", ll=.8, ul=1.2, update_on_step=uos)
        self.params.add_variable("shaprness", tag="vis", ll=.9, ul=1.1, update_on_step=uos)
        self.params.add_variable("blur", tag="vis", ll=0, ul=0.5, update_on_step=uos)
        self.params.add_variable("hue", tag="vis", ll=-.03, ul=.03, update_on_step=uos)
        # self.params.add_variable('vis/light_direction', ll=(-1, -1, 0),
        # ul=(1, 1, 1), f=lambda x: x if x[0] != 0 and x[1] != 0 else [0.01, 0.01, 1])
        self.params.add_variable("light_direction", tag="vis", ll=(-3, -3, 2), ul=(3, 3, 10),
                                 f=lambda x: x if x[0] != 0 and x[1] != 0 else [3, 3, 10])

        calib_fov = calibration["cam_fov"]
        calib_pos = calibration["cam_pos"]
        calib_orn = calibration["cam_orn"]
        self.params.add_variable("cam_fov", tag="cam", center=calib_fov, d=0.5)
        self.params.add_variable("calib_pos", tag="cam",
                                 center=calib_pos, d=(0.002, 0.002, 0.002))
        self.params.add_variable("calib_orn", tag="cam",
                                 center=calib_orn, d=(0.01, 0.01, 0.01))

    def set_cid(self, cid):
        self.cid = cid

    def get_buffer_size(self):
        return self.width, self.height

    def set_robot(self, robot):
        '''set the robot, needs to called after robot is loaded
        '''
        self.robot_uid = robot.robot_uid
        self.camera_index = robot.camera_index

    def reset(self):
        """
        Define render variables.
        """
        # TODO(lukas): set this to a good value for stacking
        self._base_view_matrix = self.p.computeViewMatrixFromYawPitchRoll(
            [0, 0, -0.2], 1.3, 0, -45, 0, 2)
        # [0.52,-0.2,-0.33], .8, 180, -41, 0, 2) debug
        if self.params.cam_fov is not None:
            fov = self.params.cam_fov
        else:
            fov = 60
        self._base_proj_matrix = self.p.computeProjectionMatrixFOV(
            fov=fov, aspect=float(self.width) / self.height,
            nearVal=self._camera_near, farVal=self._camera_far)

        fov_radians = fov * pi / 180
        f_x = self.width / (2 * tan(fov_radians/2))
        f_y = self.height / (2 * tan(fov_radians/2))
        self.camera_calibration = dict(width=self.width, height=self.height,
                                       ppx=self.width/2, ppy=self.height/2,
                                       fx=f_x, fy=f_y)
        self.zoom = self.params.zoom

    def get_view_mat(self):
        """
        This code gets called for every rendering, should be more efficient.
        """
        camera_ls = self.p.getLinkState(self.robot_uid, self.camera_index, physicsClientId=self.cid)
        camera_pos, camera_orn = camera_ls[:2]

        cam_rot = self.p.getMatrixFromQuaternion(camera_orn)
        cam_rot = np.array(cam_rot).reshape(3, 3)
        # orientation from euler
        calib_orn = self.p.getQuaternionFromEuler(self.params.calib_orn)
        calib_orn = self.p.getMatrixFromQuaternion(calib_orn)
        calib_orn = np.array(calib_orn).reshape(3, 3)
        cam_rot = cam_rot.dot(calib_orn)
        camera_pos += cam_rot.dot(self.params.calib_pos)

        cam_rot_y, cam_rot_z = cam_rot[:, 1], cam_rot[:, 2]
        # camera: eye position, target position, up vector
        view_mat = self.p.computeViewMatrix(camera_pos, camera_pos + cam_rot_y,
                                            -cam_rot_z)
        return view_mat

    @staticmethod
    def get_tcp_in_cam(T_tcp_cam):
        """
        Transform the tcp position in the camera frame into camera coordinates
        """
        # TODO(max): this should probably not have parameters hard coded
        raise NotImplementedError

        def project(k_matrix, point_x):
            '''project and normalize coordinates'''
            img_x = k_matrix @ point_x
            return img_x[0:2] / img_x[2]
        f_x = 84 / (2 * np.tan((42.82 * math.pi)/360))
        f_y = f_x
        ppx = 42
        ppy = 42
        k_matrix = np.array([[f_x, 0, ppx, 0],
                             [0, f_y, ppy, 0],
                             [0, 0, 1, 0]])
        point = project(k_matrix, np.linalg.inv(T_tcp_cam) @ np.array([0, 0.04, 0, 1]))
        return point

    def render(self, mode='rgb_array', info=None):
        """
        Render a camera image.
        """
        if mode == 'human':
            raise NotImplementedError
        if self.img_type != 'rgb':
            raise NotImplementedError

        # render
        if self.robot_cam:
            view_mat = self.get_view_mat()
        else:
            view_mat = self._base_view_matrix

        proj_mat = self._base_proj_matrix

        if self.params:
            light_direction = self.params.light_direction
        else:
            light_direction = [1, 1, 1]

        # shadows = self.params.shadows
        # TODO(max): if we want to have zoom it should be implemented
        # over the camera parameters not via some post processing stuff
        zoom_width = int(self.width * self.zoom)
        zoom_height = int(self.height * self.zoom)

        # TODO(max): add a variable to control this from outside
        flag = self.p.ER_NO_SEGMENTATION_MASK
        # else flag = self.p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX

        camera_images = self.p.getCameraImage(width=zoom_width,
                                              height=zoom_height,
                                              viewMatrix=view_mat,
                                              projectionMatrix=proj_mat,
                                              shadow=1,
                                              lightDirection=light_direction,
                                              renderer=self.p.ER_BULLET_HARDWARE_OPENGL,
                                              flags=flag,
                                              physicsClientId=self.cid)
        # hight, width, img_arr, depth_arr, obj_arr = ci
        _, _, img_arr, depth_arr, obj_arr = camera_images
        w_1 = int((zoom_width - self.width) / 2)
        h_1 = int((zoom_height - self.height) / 2)

        rgb = img_arr[h_1:h_1+self.width, w_1:w_1+self.width, :3]
        rgb = self.transform_img(rgb)

        # TODO(max): this should use get_tcp_in_cam() and add a red dot to rgb
        # show_tcp = False
        # if show_tcp:
        #    raise NotImplementedError

        # add additional inforomation to output
        if info:
            # compute correct depth image; bullet uses OpenGL to render, and the
            # convention is non-linear z-buffer, see documentation
            depth = depth_arr[h_1:h_1+self.width, w_1:w_1+self.width]
            near = self._camera_near
            far = self._camera_far
            depth = far * near / (far - (far - near) * depth)

            seg_mask = obj_arr[h_1:h_1+self.width, w_1:w_1+self.width]

            info["depth"] = depth
            info["seg_mask"] = seg_mask

        # This has shape, of e.g. (84, 84, 3)
        return rgb

    def transform_img(self, original_image):
        """
        Augment a rendered image.
        This function should only change appearance, and not geometry.
        """
        con = self.params.contrast
        bright = self.params.brightness
        col = self.params.color
        sharp = self.params.shaprness
        blur = self.params.blur
        hue = self.params.hue
        # print(con, bright, col, sharp, blur,hue)

        img = None
        if con != 1.0:
            if img is None:
                img = Image.fromarray(original_image)
            contrast = ImageEnhance.Contrast(img)
            img = contrast.enhance(con)
        if bright != 1.0:
            if img is None:
                img = Image.fromarray(original_image)
            brightness = ImageEnhance.Brightness(img)
            img = brightness.enhance(bright)
        if col != 1.0:
            if img is None:
                img = Image.fromarray(original_image)
            color = ImageEnhance.Color(img)
            img = color.enhance(col)
        if sharp != 1.0:
            if img is None:
                img = Image.fromarray(original_image)
            for _ in range(5):
                sharpness = ImageEnhance.Sharpness(img)
                img = sharpness.enhance(sharp)
        if blur != 0:
            if img is None:
                img = Image.fromarray(original_image)
            img = img.filter(ImageFilter.GaussianBlur(radius=blur))
        if hue != 0.0:
            if img is None:
                img = Image.fromarray(original_image)
            img = np.array(img) / 255
            img_hsv = matplotlib.colors.rgb_to_hsv(img)
            img_hsv[:, :, 0] += hue
            img = matplotlib.colors.hsv_to_rgb(np.clip(img_hsv, 0, 1))
            img *= 255

        if img is None:
            return original_image
        return np.array(img, dtype=np.uint8)
