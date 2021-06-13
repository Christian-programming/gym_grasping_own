"""
Unit Tests
"""
import unittest
from gym_grasping.envs.robot_sim_env import RobotSimEnv


class TestTransformImg(unittest.TestCase):
    """
    Test image transformations
    """

    @classmethod
    def setUpClass(cls):
        cls.env = RobotSimEnv()

    @classmethod
    def tearDownClass(cls):
        del cls.env

    def test_transform_func(self):
        """
        test transformations can be applied
        """
        import numpy as np
        tmp = np.random.random((12, 12, 3)).astype(np.uint8)

        self.env.params.sample["blur"] = 0.5
        self.env.params.sample["hue"] = .8
        for _ in range(2):
            self.env.camera.transform_img(tmp)


if __name__ == '__main__':
    unittest.main()
