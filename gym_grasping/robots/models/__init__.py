import os


resdir = os.path.join(os.path.dirname(__file__))


def get_path():
    return resdir


def get_robot_path(fn):
    return os.path.join(resdir, fn)
