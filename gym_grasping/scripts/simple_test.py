"""
A simple test script

Note:
Be carefull not to click on sliders while over the robot becasue mouse actions
go through the rendered panel.
"""
import pybullet as p
from gym_grasping.envs.robot_sim_env import RobotSimEnv


def debug_viewer(task_name="stack"):
    """
    Control the robot interactively by controlling sliders in the debug viewer.

    """

    # default variables
    renderer = "debug"
    robot = "kuka"
    control = "relative"
    param_info = None

    # create a new env
    env = RobotSimEnv(task=task_name, robot=robot, renderer=renderer, control=control,
                      param_info=param_info)

    # get action input from debug viewer interface
    control_names = env.robot.control_names
    defaults = [0, ]*len(control_names)
    motors_ids = []
    for ctrl_name, delta_v in zip(control_names, defaults):
        motors_ids.append(p.addUserDebugParameter(ctrl_name, -1, 1, delta_v))
    motors_ids.append(p.addUserDebugParameter("debug", -1, 1, 0))

    # main loop
    debug = False
    done = False
    while 1:
        action = []
        for motor_id in motors_ids:
            action.append(p.readUserDebugParameter(motor_id))

        debug = action[-1] > .5
        action = action[:-1]
        state, reward, done, _ = env.step(action)

        if done or debug:
            print("reward:", reward)
            env.reset()
        # img = state[:, :, ::-1]

    print("reward:", reward)


if __name__ == "__main__":
    debug_viewer()
