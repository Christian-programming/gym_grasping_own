"""
Util functions for taks.
"""
import colorsys
import os
import pybullet as p_global

from math import acos, pi

import numpy as np
GLOBAL_SCALE = 1
RESDIR = os.path.join(os.path.dirname(__file__))


def opath(file_name):
    """
    Make an absolute path out of a local one, where the local one is relative
    to the task directory.
    """
    return os.path.join(RESDIR, file_name)


def load_model(filename, pb_server, *args, **kwargs):
    ''' wrapper to load differnt model files with the same function
        the return value is a list of ids
        use_pybullet_data(bool): if True the pybullet_data path is joint to
                                 the file
    '''

    assert os.path.isfile(filename), "model not exists: {}".format(filename)
    extension = os.path.splitext(filename)[1][1:].lower()
    if extension == "urdf":
        # return a list to keep the return value same with for sd and mjcf
        return [pb_server.loadURDF(filename, *args, **kwargs)]
    elif extension == "sdf":
        return list(pb_server.loadSDF(filename, *args, **kwargs))
    elif extension == "xml":
        return list(pb_server.loadMJCF(filename, *args, **kwargs))
    else:
        raise ValueError("extension {} not supported".format(extension))


def add_debug_point(pos, orn, pb_server, cid, line_width=5, life_time=0, line_len=0.1,
                    text="", color_factor=1):
    ''' add a debug point with orientation lines to the pyblullet gui
        the orn is shown with three lines blue:x y:red z:green
        color_factor(float): change the orn xyz-lines color
        usage:
                pbh.add_debug_point([1,2,3], [0, 0, 0, 1],
                                    line_len=0.05,
                                    text="base", color_factor=1)

    '''
    startpoints = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    endpoints = np.array(
        [[line_len, 0, 0], [0, line_len, 0], [0, 0, line_len]])
    # x axis in red, y in green, z in blue
    line_color_rgb = np.array(
        [[color_factor, 0, 0], [0, color_factor, 0], [0, 0, color_factor]])
    # q_zero = np.array(p.getQuaternionFromEuler(np.array([0.,90.,0.])*(np.pi/180.)))
    q_zero = np.array([0, 0, 0, 1])
    for c, start, end in zip(line_color_rgb, startpoints, endpoints):
        start, _ = np.array(
            pb_server.multiplyTransforms(pos, orn, start, q_zero))
        end, _ = np.array(pb_server.multiplyTransforms(pos, orn, end, q_zero))
        pb_server.addUserDebugLine(lineFromXYZ=start, lineToXYZ=end,
                                   lineColorRGB=c,
                                   lineWidth=line_width,
                                   lifeTime=life_time, physicsClientId=cid)
    if text is not None or text != "":
        pb_server.addUserDebugText(text, pos, lifeTime=life_time, physicsClientId=cid)


def get_model_files(mdl_dir, file_types=(".urdf", ".sdf", ".xml")):
    ''' return sorted model files in a range in a dir '''
    cube_files = [f for f in os.listdir(
        mdl_dir) if f.strip().lower().endswith(file_types)]
    # sorted so that same on different os
    return [os.path.join(mdl_dir, mdl_f) for mdl_f in sorted(cube_files)]


_EPS = np.finfo(float).eps * 4.0


def q_dst(q0, q1):
    '''quaternion distance measure'''
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS:
        return 0.0
    else:
        return 2 * acos(abs(d)) / pi


def pose2tuple(tmp):
    '''curriculum stuff'''
    pos = tmp[:3] * GLOBAL_SCALE
    orn = np.array(p_global.getQuaternionFromEuler([0, 0, tmp[3]]))
    return pos, orn


def hsv2rgb(hsv):
    return colorsys.hsv_to_rgb(*hsv) + (1,)
