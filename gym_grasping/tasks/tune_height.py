"""
Run simulation to find rest height of objects.
"""

# add parent dir to find package. Only needed for source code build, pip install doesn't need it.
from collections import Counter, OrderedDict
from pdb import set_trace
import numpy as np
from matplotlib import pyplot as plt

from gym_grasping.envs.robot_sim_env import RobotSimEnv


def test():
    """
    Run simulation to find rest height of objects.
    """

    def robot_contacts(env):
        contacts = env.p.getContactPoints(env.robot.robot_uid, env.cid)
        contacts_uids = set([cnt[2] for cnt in contacts])
        return contacts_uids

    def summarize(count, uids):
        count_surface = Counter()
        uid_set = set(uids)
        for cnt in count:
            key, value = (bool(uid_set.intersection(cnt[0])), cnt[1])
            count_surface[key] += value
        rate = count_surface[True]/(count_surface[True]+count_surface[False])
        return rate

    def trial_env(env):
        policy = env._task.get_policy(mode='random')(env)
        env_done = False
        info_counter = []

        for _ in range(10):
            if env_done:
                env.reset()

            contacts = robot_contacts(env)
            action = policy.act(env)
            state, _, env_done, info = env.step(action)

            # we are testing starting conditions
            # reset after first iteration
            env_done = True

            if env_done:
                # info["reward"] = reward
                info = OrderedDict(info)
                contacts = tuple(contacts)
                info_counter.append(contacts)

            if env_done and "clear" in info and "first" in info:
                plt.imshow(state)
                plt.show()
                set_trace()

        return info_counter

    heights = []
    surface_rates = []
    object_rates = []

    env = RobotSimEnv(task='block', act_type='continuous', initial_pose='close',
                      renderer='debug')

    for gripper_height in (.27, .28, .29, .30):
        env._task.default_gripper_offset = np.array([0, 0, gripper_height])
        env.reset()

        print("gripper height", gripper_height)

        info_counter = trial_env(env)
        count = Counter(info_counter).most_common()
        # print("contacts")
        # for c in count:
        #    print(c[1],'\t',c[0])
        surface_rate = summarize(count, env._task.surfaces)
        print("surface_rate\t", surface_rate)

        object_rate = summarize(count, env._task.objects)
        print("object_rate\t", object_rate)

        # collect data
        heights.append(gripper_height)
        surface_rates.append(surface_rate)
        object_rates.append(object_rate)

    plt.plot(heights, surface_rates, 'o-', label="surface contacts")
    plt.plot(heights, object_rates, 'o-', label="object contacts")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test()
