"""
Test functional beahviour through built-in policies.
"""
import os
import time
import unittest
from gym_grasping.envs.robot_sim_env import RobotSimEnv

is_ci = "CI" in os.environ
renderer = "tiny" if is_ci else "debug"


class TestPickNPlace(unittest.TestCase):
    """
    Test a Pick-n-Place task.
    """

    def test_suction(self):
        """test performance of scripted policy, with suction gripper"""
        env = RobotSimEnv(task="pick_n_place", robot="suction",
                          renderer=renderer,
                          act_type='continuous', control="absolute",
                          max_steps=600, initial_pose="close")

        success_count = 0
        start_time = time.time()
        env_done = False
        for iteration in range(500):
            action, policy_done = env._task.policy(env)
            _, reward, env_done, _ = env.step(action)

            if env_done and reward > 0:
                success_count += 1
                break

        end_time = time.time()

        time_target_s = 3.0
        if env._renderer == "debug":
            time_target_s *= 2

        self.assertGreater(success_count, 0)
        self.assertLess(end_time-start_time, time_target_s)

    def test_gripper(self):
        """test performance of scripted policy, with parallel gripper"""
        env = RobotSimEnv(task="pick_n_place", robot="kuka",
                          renderer=renderer,
                          act_type='continuous', control="absolute",
                          max_steps=600, initial_pose="close")

        success_count = 0
        start_time = time.time()
        env_done = False
        for iteration in range(500):
            action, policy_done = env._task.policy(env)
            _, reward, env_done, _ = env.step(action)

            if env_done and reward > 0:
                success_count += 1
                break

        end_time = time.time()

        time_target_s = 3.0
        if env._renderer == "debug":
            time_target_s *= 2

        self.assertGreater(success_count, 0)
        self.assertLess(end_time-start_time, time_target_s)


if __name__ == '__main__':
    unittest.main()
