"""
Test functional beahviour through built-in policies.
"""
import unittest
from gym_grasping.envs.robot_sim_env import RobotSimEnv


class TestParams(unittest.TestCase):
    """
    Test a Pick-n-Place task.
    """

    def test_constant_stack(self):
        """test performance of scripted policy, with suction gripper"""

        task_name = "stack"
        env = RobotSimEnv(task=task_name, act_type='continuous',
                          renderer='tiny', initial_pose='close')
        params = env.params.variables

        import pickle

        # with open( "stack_params.pkl", "wb" ) as f_obj:
        #    pickle.dump(params,  f_obj)

        with open("stack_params.pkl", "rb") as f_obj:
            params_old = pickle.load(f_obj)

        for k, v in params_old.items():
            assert k in params
            v2 = params[k]
            try:
                del v["f"]
            except KeyError:
                pass
            try:
                del v2["f"]
            except KeyError:
                pass
            self.assertEqual(v, v2)


if __name__ == '__main__':
    unittest.main()
