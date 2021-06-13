"""
Unit Tests
"""
import time
import unittest
# before we do the imports, make sure to cache which envs we already have
# so that we get a set of envs from gym_grasping
from gym.envs.registration import registry
ENVS_ORIG = set(registry.all())
from gym_grasping.envs.robot_sim_env import RobotSimEnv
ENVS_NEW = set(registry.all())


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

            env = env_spec.make()
            for _ in range(3):
                action = env.action_space.sample()
                env.step(action)
            del env
            # maybe this helps OpenGL to vacate the contexts
            time.sleep(.1)


# class TestSampler(unittest.TestCase):
#    """
#    Test parameter sampler.
#    """
#    @classmethod
#    def setUpClass(cls):
#        cls.env = AdaptiveCurriculumEnv(task="stackVel", curr="stack")

#    @classmethod
#    def tearDownClass(cls):
#        del cls.env

#    def test_default_params(self):
#        """
#        test sampled augmentations are different.
#        """
#        self.env.reset(data=dict(difficulty=1, load_demo=False))
#        con = self.env.params.contrast
#        bright = self.env.params.brightness
#        col = self.env.params.color
#        sharp = self.env.params.shaprness
#        blur = self.env.params.blur
#        hue = self.env.params.hue

#        self.env.reset(data=dict(difficulty=1, load_demo=False))
#        con2 = self.env.params.contrast
#        bright2 = self.env.params.brightness
#        col2 = self.env.params.color
#        sharp2 = self.env.params.shaprness
#        blur2 = self.env.params.blur
#        hue2 = self.env.params.hue

#        self.assertNotEqual(con, con2)
#        self.assertNotEqual(bright, bright2)
#        self.assertNotEqual(col, col2)
#        self.assertNotEqual(sharp, sharp2)
#        self.assertNotEqual(blur, blur2)
#        self.assertNotEqual(hue, hue2)

# class ProfileBlockEnv(unittest.TestCase):
#    """
#    Test env performance
#    """

#    def test_perf(self):
#        """test performance of random policy, no rendering"""
#        env = RobotSimEnv(task='block', act_type='continuous',
#                          initial_pose='close',
#                          obs_type='state')

#        success_count = 0
#        env_count = 0
#        start_time = time.time()
#        env_done = False
#        for iteration in tqdm(range(2000)):
#            action = env.action_space.sample()
#            _, reward, env_done, info = env.step(action)
#            if env_done and reward > 0:
#                success_count += 1

#            if env_done:
#                env.reset()
#                env_count += 1

#            if env_done:
#                self.assertFalse("clear" in info and "first" in info)

#            if iteration % 100 == 0 and iteration > 0:
#                print("FPS: ", 100 / (time.time() - start_time))
#                start_time = time.time()

#        self.assertGreater(success_count/(iteration+1), .001)
#        self.assertGreater(success_count/env_count, .05)


if __name__ == '__main__':
    unittest.main()
