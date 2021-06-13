"""
Unit Tests
"""
import unittest
# before we do the imports, make sure to cache which envs we already have
# so that we get a set of envs from gym_grasping
from gym.envs.registration import registry
ENVS_ORIG = set(registry.all())
from gym_grasping.envs.robot_sim_env import RobotSimEnv
ENVS_NEW = set(registry.all())


class TestDefaultEnv(unittest.TestCase):
    """
    Test creation of default env
    """
    @classmethod
    def setUpClass(cls):
        cls.env = RobotSimEnv()

    @classmethod
    def tearDownClass(cls):
        del cls.env

    def test_robot_control_type(self):
        '''test default robot control type'''
        self.assertEqual(self.env.robot.discrete_control, False)

    def test_gripper_control_type(self):
        '''test default gripper action type'''
        self.assertEqual(self.env.robot.gripper.act_type, 'discrete')


class TestNamedEnvs(unittest.TestCase):
    """
    Test creation of all named envs.
    """

    def test_more_envs(self):
        '''test that we registered envs'''
        self.assertGreater(len(ENVS_NEW), len(ENVS_ORIG))

    @staticmethod
    def test_envs():
        """
        test envs can be created.
        """
        envs_diff = ENVS_NEW.difference(ENVS_ORIG)
        for env_i, env_spec in enumerate(envs_diff):

            print(env_i, env_spec)
            # overwrite the renderer EGL/debug->tiny for CI machines
            if 'renderer' in env_spec._kwargs and env_spec._kwargs['renderer'] in ('egl', 'debug'):
                print("Warning: overiting default renderer to 'tiny'")
                env_spec._kwargs['renderer'] = 'tiny'

            env = env_spec.make()
            for _ in range(3):
                action = env.action_space.sample()
                env.step(action)
            del env


if __name__ == '__main__':
    unittest.main()
