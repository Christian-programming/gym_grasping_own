import numpy as np


class Play:
    def __init__(self, env, action=None, pressed_keys=None):
        pass

    def get_defaults(self):
        return None

    def act(self, env, action):
        # env.reset()
        return action


class GraspHeuristicClose:
    def __init__(self, env, action=None, pressed_keys=None):
        self.reset()

    def reset(self):
        # maxDist = self.robot.gripper.maxDist
        self.terminating = True
        self.terminating_sep = 0
        self.terminating_close = 0
        self.terminating_up = 0
        self.terminating_tot = 0

    def get_defaults(self):
        return None

    def act(self, env, isDiscrete=True, print_fps=False):
        numSep = 1
        numClose = 20
        numUp = 30
        graspAction = None
        if self.terminating_sep < numSep:
            graspAction = env._robot.gripper.graspActionU()
            self.terminating_sep += 1
        elif self.terminating_close < numClose:
            i = float(self.terminating_close)
            graspAction = env._robot.gripper.graspActionA(i / numClose)
            self.terminating_close += 1
        elif self.terminating_up < numUp:
            i = float(self.terminating_up)
            graspAction = env._robot.gripper.graspActionB(i / numUp)
            self.terminating_up += 1
        elif not self.terminating_up < numUp:
            self.terminated = True

        if graspAction is None:
            graspAction = env._robot.gripper.graspActionU()
        graspAction = np.array(graspAction) * 1. / .005
        return graspAction
