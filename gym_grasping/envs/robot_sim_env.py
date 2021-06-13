"""
robot_sim_env.py handles the gym-spaces api, the physics engine initialization
and task interop.
"""
import sys
import json
import pkgutil

import numpy as np
import pybullet as p
import gym
from gym import spaces
from gym.utils import seeding
from gym_grasping.envs.env_param_sampler import EnviromentParameterSampler
from gym_grasping.envs.camera import PyBulletCamera
from gym_grasping.robots.robot_names import robot_names
from gym_grasping.tasks.task_names import task_names
from gym_grasping.robots.models import get_robot_path as rpath


def obs_to_observation_space(obs):
    """
    Turn an example observation into an observation space.
    """
    if isinstance(obs, dict):
        tmp = [(k, obs_to_observation_space(v)) for k, v in obs.items()]
        space = spaces.Dict(dict(tmp))
    elif isinstance(obs, np.ndarray):
        if obs.dtype.type == np.float64:
            low, high, dtype = -np.inf, np.inf, 'float32'
        elif obs.dtype.type == np.uint8:
            low, high, dtype = 0, 255, 'uint8'
        space = spaces.Box(low=low, high=high, shape=obs.shape, dtype=dtype)
    else:
        raise NotImplementedError
    return space


class RobotSimEnv(gym.Env):
    """
    RobotSimEnv class, inherits from gym.
    This class sets up and coordinates robots, rendering and tasks, little
    logic happens here. It's still quite large because these things have many
    parameters.

    Args:
        param_info: a dictionary of parameters to overwrite the in code defs.
    """

    def __init__(self,
                 robot='kuka',
                 task='grasp',
                 seed=0,
                 # robot config
                 act_type='continuous',  # continuous discrete
                 camera_pos='new_mount',  # new_mount, old_mount, fixed, video
                 control='relative',
                 # obs config
                 obs_type='image',  # None, state, image, image_state, image_state_reduced
                 img_type='rgb',  # rgb, depth, rgbd
                 img_size='rl',  # rl, video
                 renderer='tiny',  # tiny, egl, debug, None
                 # scene setup
                 initial_pose='close',
                 # training and sim stuff
                 is_enable_self_collision=True,
                 transition_callback=None,
                 terminal_callback=None,
                 log=False,
                 sample_params=True,
                 calibration=None,
                 param_randomize=True,
                 param_info=None,
                 **kwargs):
        
        # pybullet has global scope, var in case this changes
        self.p = p
        self.gs = 1
        #
        # variable declaration, for lint
        self._task = None
        self.robot = None
        self.np_random = None
        self._observation = None
        self._observation_state = None
        self._info = None
        #
        # vars, set in reset
        self._max_steps = None
        self._default_max_steps = None
        self._ep_step_counter = None
        self._ep_counter = None
        self.zoom = None

        #
        # Parameter, see Readme
        self.params = EnviromentParameterSampler(self.np_random, init_args=kwargs)
        self.param_randomize = param_randomize

        #
        # Calibration files
        # parameter precedence is: source code < calib file < runtime overide with param_info
        # calibration files are per-robot
        # calibration files themselves have the precdence: default in src <  runtime override
        if calibration is None:
            # try getting robot calibration
            try:
                robot_calibration = robot_names[robot].calibration
            except AttributeError:
                robot_calibration = None

            if robot_calibration is not None:
                if isinstance(robot_calibration, str):
                    with open(rpath(robot_calibration), "rb") as f_obj:
                        calibration = json.load(f_obj)
                else:
                    # could be dict too
                    calibration = robot_calibration
                print("XXXX", calibration["datetime"])
        # else use init variable calibration

        #
        # Camera Setup
        # robot_cam: camera attached to robot
        if camera_pos in ("fixed", "video"):
            robot_cam = False
        elif camera_pos in ("new_mount", "old_mount"):
            robot_cam = True
        else:
            raise ValueError("Render view unknown", camera_pos)
        self.camera = PyBulletCamera(p=p, env_params=self.params,
                                     img_type=img_type, img_size=img_size,
                                     robot_cam=robot_cam, calibration=calibration)
        #
        # Create pybullet simulation
        # To optimize pybullet render speeds, tell it which maximum image
        # size to use, this must be done during the env creation
        render_width, render_height = self.camera.get_buffer_size()
        self._renderer = renderer
        if self._renderer == "tiny":
            optionstring = '--width={} --height={}'.format(
                render_width, render_height)
            cid = p.connect(p.DIRECT, options=optionstring)
        elif self._renderer == "egl":
            optionstring = '--width={} --height={}'.format(
                render_width, render_height)
            # optional --window_backend and --render_device, not implemented
            # yet
            cid = p.connect(p.DIRECT, options=optionstring)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(
                p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            egl = pkgutil.get_loader('eglRenderer')
            print("Loading EGL plugin (may segfault on misconfigured systems)")
            if egl:
                plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
            else:
                plugin = p.loadPlugin("eglRendererPlugin")
            if plugin < 0:
                print("\nPlugin Failed to load!\n")
                sys.exit()
            print("..done.")
        elif self._renderer == "debug":
            cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
        elif self._renderer is None:
            cid = p.connect(p.DIRECT)
        else:
            raise ValueError
        # check that connection is made
        if cid < 0:
            sys.exit()
        self.cid = cid
        self.camera.set_cid(cid)

        #
        # Robot and Task Setup
        self._task = task_names[task](cid, self.np_random, p, self.params,
                                      **kwargs)
        self.robot = robot_names[robot](cid, p, env_params=self.params,
                                        act_type=act_type, camera_pos=camera_pos,
                                        calibration=calibration, control=control,
                                        **kwargs)
        self.seed(seed)  # do this after robot & task init

        #
        # Control Input
        self.action_space = self.robot.action_space
        self.control = control

        #
        # Simulation variables
        self._ep_counter = 0
        self._time_step = 1. / 240.
        self._enable_self_collision = is_enable_self_collision
        self.initial_pose = initial_pose
        self.obs_type = obs_type

        self.params.add_variable("max_steps", 50, tag="sim")
        self.params.add_variable("frameskip", 4, tag="sim", f=int)
        restitution = 0.5
        self.params.add_variable("restitution", tag="sim", ll=0, ul=restitution,
                                 mu_s=0, mu_e=restitution,
                                 r_s=0.1, r_e=0.1)
        # finish param sampler so that we can load env
        self.params.init(sample_params=sample_params,
                         param_info=param_info)
        #
        # Output
        self.log = log
        # TODO(lukas): what does this do, where is it set?
        self.eval = False
        #
        # checks
        obs_type_options = (None, 'state', 'image', 'image_state',
                            'image_state_reduced')
        assert obs_type in obs_type_options, obs_type
        camera_pos_options = ("fixed", "video", "new_mount", "old_mount")
        assert camera_pos in camera_pos_options, camera_pos
        assert renderer in ('tiny', 'egl', 'debug')

        #
        # Load Env
        self._load_env()
        obs = self.reset()
        self.observation_space = obs_to_observation_space(obs)
        self._transition_callback = transition_callback
        self._terminal_callback = terminal_callback

    def __del__(self):
        """try to disconnect, in a test case the server could be not started"""
        try:
            p.disconnect(physicsClientId=self.cid)
        except p.error as error:
            print(error)

    def seed(self, seed=None):
        """
        Ideally, randomness comes from the parameter sampler
        """
        self.np_random, seed = seeding.np_random(seed)
        self._env_id = seed
        self.params.np_random = self.np_random
        if self._task is not None:
            self._task.np_random = self.np_random
        if self.robot is not None:
            self.robot.np_random = self.np_random
        return [seed]

    def _load_env(self):
        """
        Load environment (initialy run once per env, then every 50 episodes)

        This is like a hard reset, done before reset
        """
        p.resetSimulation(physicsClientId=self.cid)
        p.setPhysicsEngineParameter(numSolverIterations=150, physicsClientId=self.cid)
        p.setTimeStep(self._time_step, physicsClientId=self.cid)
        p.setGravity(0, 0, -10 * self.gs, physicsClientId=self.cid)
        # load task objects, then robot
        self._task.load_scene()
        self.robot.load(workspace_offset=self._task.robot_workspace_offset)
        self.camera.set_robot(self.robot)

    def reset(self, initial_position=None):
        """
        Reset the enviroment. This is run once per episode.

        Args:
            data: the recipe for constructing env, array of float random
                  numbers passed on to env_params.update_on_reset

        Returns:
            obs: the initial observation

        """
        self._max_steps = self.params.max_steps
        self._default_max_steps = self.params.max_steps
        self._ep_step_counter = 0
        self._ep_counter += 1
        if self._ep_counter % 100 == 0:
            self._load_env()

        # randomize configuration
        self.params.step(randomize=self.param_randomize)
        # load variable objects
        self._task.reset()
        self.robot.reset()
        self.camera.reset()

        # robot init sets fixed pose by default
        if self.initial_pose == "fixed":
            # TODO(max) reset_pose calls gripper reset, so it's called twice,
            # clean up
            self.robot.reset_pose()
            p.stepSimulation(physicsClientId=self.cid)
        elif self.initial_pose == "close":
            # This is a hacky version to allow close initalization, but take
            # a step back in case we are touching the object
            # This logic is not in task because it requires stepping sim
            target_pose = self._task.robot_target_pose()
            if self.params:
                self.robot.reset_pose(target_pose,
                                      angle=self.params.gripper_rot)
            else:
                self.robot.reset_pose(target_pose, angle=0)
            p.stepSimulation(physicsClientId=self.cid)  # reset required for pose update
            if not self._task.robot_clear(self):
                target_pose = self._task.robot_target_pose_clear()
                self.robot.reset_pose(target_pose)
                p.stepSimulation(physicsClientId=self.cid)
            else:
                pass
        elif self.initial_pose == "above":
            target_pose = self._task.robot_target_pose()
            self.robot.reset_pose(target_pose)
        elif self.initial_pose == "calib":
            self.robot.reset_pose(initial_position)
        else:
            msg = "Unknow pose was: {}, should be 'fixed','above', 'close' " \
                  "or 'calib'"
            raise ValueError(msg.format(self.initial_pose))

        # _info is *not* cleared at the end of each step, its populated by
        # _get_obs
        self._info = {}
        self._observation = self._get_obs()
        if not self._task.robot_clear(self):
            # print("Warning Env: robot not clear")
            self._info["clear"] = False
        self.state_vector = self._task.state_vector
        return self._observation

    def _create_robot_observation(self):
        '''proprioceptive robot state'''
        remaining_episode_length = ((self._max_steps - self._ep_step_counter) /
                                    self._default_max_steps)
        robot_state = np.array([*self.robot.get_observation(),
                                remaining_episode_length])
        self._info["robot_state_full"] = robot_state.copy()
        if self.obs_type == 'image_state_reduced':
            # normalize manually
            rot_ll = self.robot.rot_limits[0]
            rot_ul = self.robot.rot_limits[1]
            robot_state[2] -= 0.13
            robot_state[2] /= 0.17
            robot_state[10] = (robot_state[10] - rot_ll) / (rot_ul - rot_ll)
            robot_state[11] /= 0.07
            robot_state = robot_state[[2, 10, 11, 12]]
        return robot_state

    def render(self, mode='rgb_array'):
        '''render is gym compatibility, but done by camera'''
        return self.camera.render(mode=mode, info=self._info)

    def _get_obs(self):
        '''combine proprioceptive robot state + task state + image'''
        # process state first
        robot_state = self._create_robot_observation()
        task_state = self._task.get_state()
        if self.obs_type == 'image_state_reduced':
            task_state = task_state[:6]
        self._observation_state = np.array([*robot_state, *task_state])
        # then image
        if self.obs_type == 'state':
            self._observation = self._observation_state
        elif self.obs_type == 'image':
            self._observation = self.render()
        elif self.obs_type in ['image_state', 'image_state_reduced']:
            self._observation = {'img': self.render(),
                                 'robot_state': robot_state,
                                 'task_state': task_state}
        else:
            self._observation = None
        return self._observation

    def _termination(self):
        '''check termination'''
        over_max = self._ep_step_counter >= self._max_steps
        terminate_steps = self._max_steps is not None and over_max
        if terminate_steps:
            # print( "Termination" + [" due to episode length",
            # ". {}".format(self._reward())][self.terminated])
            return True
        return False

    def step(self, action):
        '''Step the world.'''
        self.params.update_on_step()
        self._ep_step_counter += 1  # num_steps
        #
        # frameskip
        #print("step", self._task.params.variables.keys())
        #print("step", self._task.params.variables["object_pose"])
        #print("step", self._task.params.variables["object_to_gripper"])
        #print("step", type(self._task.params))
        frameskip = int(self.params.frameskip)
        for _ in range(frameskip):
            self.robot.apply_action(action)
            p.stepSimulation(physicsClientId=self.cid)
        env_done = self._termination()
        if self.eval:
            _, reward, task_done, task_info = self._task.eval_step(self,
                                                                   action)
        else:
            _, reward, task_done, task_info = self._task.step(self, action)
        done = env_done or task_done
        if done and self._ep_step_counter == 1:  # num_steps:
            print("Warning: Env done on first iteration")
            self._info["first"] = True
        #
        # outputs
        obs = self._get_obs()
        if self._transition_callback is not None:
            self._transition_callback(self)
        if done and self._terminal_callback is not None:
            self._terminal_callback(self)
        # merge in favor of env info
        info = dict(task_info, **self._info)
        # print(self._task.state_vector)
        self.state_vector = self._task.state_vector
        # self._info = {}
        return obs, reward, done, info
