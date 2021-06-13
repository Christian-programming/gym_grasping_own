import os
import glob
import multiprocessing
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np

# env is in a different process
ENV_DATA_QUEUE = multiprocessing.Queue()


class Viewer:
    """
    A Viewer class that plots values over time.
    """
    def __init__(self, transpose=True, zoom=None, video=False):
        self.env_initialized = False
        self.run_initialized = False
        self.transpose = transpose
        self.zoom = zoom

        if video:
            os.makedirs('./video', exist_ok=True)
            files = glob.glob('./video/*.png')
            for f in files:
                os.remove(f)
            self.frame_count = 0
        self.video = video

        self.col_data = None
        self.col_data = None
        self.col_screen = None
        self.net_screen = None
        self.run_data = None

        plt.ion()
        self.fig = plt.figure(figsize=(2, 2))
        gs = gridspec.GridSpec(2, 2)
        gs.update(wspace=0.001, hspace=0.001)  # set the spacing between axes.

        self.col_ax = plt.subplot(gs[0, 0])
        self.net_ax = plt.subplot(gs[0, 1])
        self.plt_ax = plt.subplot(gs[1, :])
        self.col_ax.set_axis_off()
        self.net_ax.set_axis_off()
        self.plt_ax.set_axis_off()

        plt.subplots_adjust(wspace=0.5, hspace=0, left=0, bottom=0, right=1, top=1)

        # time series
        num_plots = 2
        self.horizon_timesteps = 30 * 5
        self.t = 0
        self.cur_plot = [None for _ in range(num_plots)]
        self.data = [deque(maxlen=self.horizon_timesteps) for _ in range(num_plots)]

    @staticmethod
    def env_callback(env):
        """
        callback for env, why is this needed, maybe multiprocessing.
        """
        obs = env._observation
        ENV_DATA_QUEUE.put(obs)

    def run_callback(self, prev_obs, obs, actions, rew, masks, values):
        """
        callback which adds new data
        """
        self.run_data = obs.copy()
        if rew in (0.0, 1.0):
            print("Run callback:", rew, masks)

        # time series
        def data_callback(prev_obs, obs, actions, rew, masks, values):
            if rew[0] == 0.0:
                rew = -.1
            elif rew[0] == 1.0:
                rew = 1
            else:
                rew = 0

            return [rew, values[0], ]
        points = data_callback(prev_obs, obs, actions, rew, masks, values)
        for point, data_series in zip(points, self.data):
            data_series.append(point)
        self.t += 1
        xmin, xmax = max(0, self.t - self.horizon_timesteps), self.t
        plot_data = zip(self.cur_plot, ['C1', 'C2'], ['rew', 'val'])
        for i, (plot, c, l) in enumerate(plot_data):
            if plot is not None:
                plot.remove()
            self.cur_plot[i], = self.plt_ax.plot(range(xmin, xmax),
                                                 list(self.data[i]),
                                                 color=c, label=l)
            self.plt_ax.set_xlim(xmin, xmax)
        self.plt_ax.legend(loc='lower left')

        self.draw()

    def draw(self):
        """
        draw updated version of the plot
        """
        col_data = ENV_DATA_QUEUE.get()
        if self.env_initialized:
            self.col_screen.set_data(col_data)
        elif col_data is not None:
            obs = col_data
            self.col_screen = self.col_ax.imshow(obs, aspect='auto')
            self.env_initialized = True

        if self.run_initialized:
            self.net_screen.set_data(self.run_data)
        elif self.run_data is not None and not np.all(self.run_data == 0):
            obs = self.run_data
            self.net_screen = self.net_ax.imshow(obs, cmap='gray', aspect='auto')
            self.run_initialized = True

        self.fig.tight_layout()
        self.fig.canvas.draw()

        if self.video:
            filename = './video/{0:03d}.png'.format(self.frame_count)
            self.fig.savefig(filename, pad_inches=0)
            self.frame_count += 1
