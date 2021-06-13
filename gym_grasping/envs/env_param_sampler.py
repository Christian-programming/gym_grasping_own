"""
Sample Environment Parameters

See project README for how parameters are managed.

We implement a curriculum mechanism that creates a schedule for increasing task
difficulty. For this we have choses a centralized mechanism for managining
parameters that are randomized. This is done via the `params` varaible
that is defined in the `Env.__init__` funciton and passed to those of its
member classes that require it, e.g. robot, camera and task.
"""
from collections import OrderedDict
import numpy as np
from gym.utils import seeding


def interp(start, stop, transition):
    '''interpolate'''
    return (1 - transition) * start + transition * stop


class EnviromentParameterSampler:
    """Parameter collection class"""

    def __init__(self, np_random, init_args=None, variables=None,
                 max_difficulty=1.0):
        """
        Enviroment Parameter Sampler

        for now this is based on uniform sampling x in [0,1] with
        x' = f(Ax+B). variable name is set to everything after

        As a convention please use same variable names as in classes.
        e.g. env.robot = Robot(dv=.2) -> "robot_dv"

        Args:
            np_random: numpy random state
        """

        self.np_random = np_random
        self.variables = OrderedDict()
        self.sample_func = {}
        self.sample = {}
        self.lock = False
        self.max_possible_difficulty = max_difficulty
        self.max_difficulty = max_difficulty
        if init_args is not None:
            pass
        else:
            init_args = {}

        self.init_args = init_args

        if variables is not None:
            self.variables = variables
            self.init()

    def __getattr__(self, name):
        try:
            return self.sample[name]
        except KeyError:
            print("Variable {} not defined in param sampler".format(name))
            raise AttributeError

    def add_variable(self, name, *args, tag=None, **kwargs):
        '''add a variable for the first time'''
        if self.lock:
            raise ValueError
        if name in self.variables:
            print("Warning: parameter {} already defined".format(name))
        # helper functions
        if len(args) == 1:
            kwargs = dict(center=args[0], d=0)
        if "center" in kwargs and "d" not in kwargs:
            kwargs["d"] = 0
        if "f" in kwargs:
            kwargs["f"] = CloudpickleWrapper(kwargs["f"])
        self.variables[name] = kwargs

    def set_sample(self, name, value):
        '''overwrite the sampling'''
        self.sample_func[name] = value

    def update_on_step(self):
        '''update for each step. Deprecated'''
        uniform = self.np_random.uniform(-1, 1, self.num_samples).astype(np.float32)
        # _sample = _A*x + _B
        self._sample[self.update_on_step_ids] = np.clip(
            interp(self._MU_S, self._MU_E, self._DIFF_MU) +
            interp(self._R_S, self._R_E, self._DIFF_R) * uniform, self._LL,
            self._UL)[self.update_on_step_ids]
        for var_id in self.sample_func:
            self.sample[var_id] = self.sample_func[var_id]()

    def sample_specific(self, name):
        '''sample a specific variable?'''
        shortname = name.split("/")[-1]
        # print("Sample specific", name)
        var_id = self.variable_ids[name]
        uniform = self.np_random.uniform(-1, 1, self.num_samples).astype(np.float32)
        # _sample = _A*x + _B
        self._sample[var_id] = np.clip(
            interp(self._MU_S, self._MU_E, self._DIFF_MU) +
            interp(self._R_S, self._R_E, self._DIFF_R) * uniform, self._LL,
            self._UL)[var_id]
        self.sample[shortname] = self.sample_func[shortname]()

    def get_curriculum_info(self):
        'get the variable info as a dict'''
        curriculum_info = {}
        for name in self.variables.keys():
            curriculum_info[name] = getattr(self, name.split("/")[-1])
        return curriculum_info

    def init(self, sample_params=True, param_info=None):
        '''This setattr stuff is a bit hacky, maybe replace with dict?'''
        if sample_params:
            self.max_difficulty = self.max_possible_difficulty
        else:
            self.max_difficulty = 0

        if param_info is not None:
            var_keys = self.variables.keys()
            par_keys = param_info.keys()
            assert set(var_keys).issuperset(set(par_keys))
            self.variables.update(param_info)

        # prevent addtional variables from being added
        self.lock = True
        i = 0
        LL = []
        UL = []
        DIFF_MU = []
        DIFF_R = []
        MU_S = []
        MU_E = []
        R_S = []
        R_E = []
        self.variable_ids = {}
        self.update_on_step_ids = []
        used_names = ["num_samples", "step", "init", "add_variable"]

        for var, kwargs in self.variables.items():
            name = var.split("/")[-1]
            if name in used_names:
                raise ValueError

            if name in self.init_args:
                try:
                    print("param {}: {} -> {}".format(name, kwargs["center"], self.init_args[name]))
                except KeyError:
                    print("XXXXXXXXXXXXXXXXXX", name)
                ul = ll = self.init_args[name]
            elif "center" in kwargs and isinstance(kwargs["center"], (int, float)):
                ll = kwargs["center"] - kwargs["d"]
                ul = kwargs["center"] + kwargs["d"]
            elif "center" in kwargs:
                ll = [c-d for c, d in zip(kwargs["center"], kwargs["d"])]
                ul = [c+d for c, d in zip(kwargs["center"], kwargs["d"])]
            else:
                ul = kwargs["ul"]
                if "ll" in kwargs:
                    ll = kwargs["ll"]
                else:
                    ll = None

            if "mu_s" in kwargs:
                if "mu_e" not in kwargs:
                    err_msg = "if mu_s is given, mu_e must be given too"
                    raise ValueError(err_msg)
                mu_s = kwargs['mu_s']
                mu_e = kwargs['mu_e']
            else:
                if ll is None:
                    err_msg = "if not mu_s and_e is given, ll must be given"
                    raise ValueError(var, err_msg)
                if isinstance(ul, (int, float)):
                    mu_s = mu_e = (ul + ll) / 2
                else:
                    mu_s = mu_e = [(u + l) / 2 for u, l in zip(ul, ll)]

            if "r_s" in kwargs:
                if "r_e" not in kwargs:
                    raise ValueError("if r_s is given, r_e must be given too")
                r_s = kwargs['r_s']
                r_e = kwargs['r_e']
            else:
                if ll is None:
                    err_msg = "if not mu_s and_e is given, ll must be given"
                    raise ValueError(err_msg)
                if isinstance(ul, (int, float)):
                    r_s = 0
                    r_e = (ul - ll) / 2
                else:
                    r_s = [0 for _ in ul]
                    r_e = [(u - l) / 2 for u, l in zip(ul, ll)]

            if "f" in kwargs:
                # recover function from CloudpickleWrapper
                # I think this chnanges between python3.5 and 3.7
                try:
                    f = kwargs["f"].x
                except AttributeError:
                    f = kwargs["f"]
            else:
                def f(x):
                    return x

            if not isinstance(ul, (list, tuple)):
                if ll is None:
                    ll = 0
                # A.append((ul - ll)/2)
                # B.append((ll + ul)/2)
                UL.append(ul)
                LL.append(ll)
                DIFF_MU.append(self.max_difficulty)
                DIFF_R.append(self.max_difficulty)
                MU_S.append(mu_s)
                MU_E.append(mu_e)
                R_S.append(r_s)
                R_E.append(r_e)

                def thunk(name, i, f):
                    '''thunk function'''
                    return lambda: f(self._sample[i])
                self.set_sample(name, thunk(name, i, f))
                self.variable_ids[var] = i
                if "update_on_step" in kwargs and kwargs['update_on_step']:
                    self.update_on_step_ids.append(i)
                i += 1
            else:
                l_num = len(ul)
                if ll is None:
                    ll = tuple(0 for _ in range(l_num))
                elif len(ll) != l_num:
                    ll = tuple(ll for _ in range(l_num))
                for j in range(l_num):
                    # A.append((ul[j]-ll[j])/2)
                    # B.append((ll[j]+ul[j])/2)
                    UL.append(ul[j])
                    LL.append(ll[j])
                    DIFF_MU.append(self.max_difficulty)
                    DIFF_R.append(self.max_difficulty)
                    MU_S.append(mu_s[j])
                    MU_E.append(mu_e[j])
                    R_S.append(r_s[j])
                    R_E.append(r_e[j])
                    if "update_on_step" in kwargs and kwargs['update_on_step']:
                        self.update_on_step_ids.append(i+j)

                def thunk(name_, i_, l_num_, f_):
                    '''thunk function'''
                    return lambda: f_(self._sample[i_:i_+l_num_])

                self.set_sample(name, thunk(name, i, l_num, f))
                self.variable_ids[var] = list(range(i, i + l_num))
                i += l_num

            used_names.append(name)
            del ll, ul, r_s, r_e, mu_s, mu_e

        self.num_samples = i
        # self._A = np.array(A)
        # self._B = np.array(B)
        self._UL = np.array(UL)
        self._LL = np.array(LL)
        self._MU_S = np.array(MU_S)
        self._MU_E = np.array(MU_E)
        self._R_S = np.array(R_S)
        self._R_E = np.array(R_E)
        self._DIFF_MU = np.array(DIFF_MU)
        self._DIFF_R = np.array(DIFF_R)

        # make a first psedo step, so that we can init objects to right
        # positions
        self.step()

    def set_variable_difficulty_mu(self, name, diff):
        '''set the difficulty for single variable'''
        assert 0 <= diff <= 1
        shortname = name.split("/")[-1]
        i = self.variable_ids[shortname]
        self._DIFF_MU[i] = diff

    def set_variable_difficulty_r(self, name, diff):
        '''set the difficulty for single variable'''
        assert 0 <= diff <= 1
        shortname = name.split("/")[-1]
        i = self.variable_ids[shortname]
        self._DIFF_R[i] = diff

    def step(self, randomize=True):
        """
        sample new random numbers, then generate randomized variables.
        This is usually called when the env is reset.

        Args:
            randomize: turns variable randomization on or off.
        """

        if randomize:
            uniform = self.np_random.uniform(-1, 1, self.num_samples)
        else:
            uniform = np.zeros(self.num_samples)
        uniform = uniform.astype(np.float32)
        assert uniform.shape == (self.num_samples,)

        # self._sample = self._A*x + self._B
        tmp_a = interp(self._MU_S, self._MU_E, self._DIFF_MU)
        tmp_b = interp(self._R_S, self._R_E, self._DIFF_R)
        sample = tmp_a + tmp_b * uniform
        self._sample = np.clip(sample, self._LL, self._UL).astype(np.float32)

        # next turn random number samples into dict
        for var_id in self.sample_func:
            self.sample[var_id] = self.sample_func[var_id]()


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


def test_param_sampler():
    '''
    test the param sampler
    '''
    seed = seeding.np_random()[0]
    param_sampler = EnviromentParameterSampler(np_random=seed)
    param_sampler.add_variable("dyn/robot_dv", center=.001, d=.0002)
    param_sampler.add_variable("a/b", center=.001, d=.0002)
    param_sampler.init()
    param_sampler.set_variable_difficulty_mu("dyn/robot_dv", 0.005)
    param_sampler.set_variable_difficulty_r("dyn/robot_dv", 0.005)
    print(param_sampler.robot_dv)
    param_sampler.reset()
    print(param_sampler.robot_dv)
    print(param_sampler.robot_dv)
    param_sampler.reset()
    print(param_sampler.robot_dv)


if __name__ == "__main__":
    test_param_sampler()
