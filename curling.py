"""
Curling task
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np


class CurlingEnv(gym.Env):
    """
    Description:
        A curling is moving around the ground

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Curling x coordinate      1                       99
        1       Curling y coordinate      1                       99
        2       Curling x velocity        -Inf                    Inf
        3       Curling y velocity        -Inf                    Inf

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push curling to the up
        1     Push curling to the right
        2     Push curling to the left
        3     Push curling to the down

    Reward:
        Distance from the target point

    Starting State:
        Curling , whose velocity is set to [-10, 10] randomly, appears at any point on the ground at the beginning

    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.mass, self.radius = 1, 1
        self.res_coef = 0.0005
        self.x_threshold = 100
        self.action_cooldown = 0
        self.current_action = 4
        self.seconde_per_step = 0.01

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array([self.x_threshold - self.radius, self.x_threshold - self.radius,
                         np.finfo(np.float32).max, np.finfo(np.float32).max],
                        dtype=np.float32)

        low = np.array([self.radius, self.radius, np.finfo(np.float32).max, np.finfo(np.float32).max], dtype=np.float32)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.target_points = None

        self.seed()
        self.viewer = None
        self.state = None

        self.curling, self.curling_transform = None, None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        x, y, x_dot, y_dot = self.state

        if self.action_cooldown == 0:
            self.current_action = action
            self.action_cooldown = 9
        else:
            self.action_cooldown -= 1

        if self.current_action == 0:
            x_force, y_force = 5, 0
        elif self.current_action == 1:
            x_force, y_force = 0, 5
        elif self.current_action == 2:
            x_force, y_force = -5, 0
        elif self.current_action == 3:
            x_force, y_force = 0, -5
        else:
            raise ValueError('Wrong action for curling!')

        if x_dot > 0:
            x_force -= self.res_coef * (x_dot ** 2)
        elif x_dot < 0:
            x_force += self.res_coef * (x_dot ** 2)

        if y_dot > 0:
            y_force -= self.res_coef * (y_dot ** 2)
        elif y_dot < 0:
            y_force += self.res_coef * (y_dot ** 2)

        next_x, next_y, x_dot, y_dot = self.__step(x=x, y=y, x_dot=x_dot, y_dot=y_dot,
                                                   x_force=x_force, y_force=y_force)

        self.state = (next_x, next_y, x_dot, y_dot)

        done = False
        reward = -self._distance(x=next_x, y=next_y)

        return np.array(self.state), reward, done, {'cooldown': self.action_cooldown}

    def reset(self):
        self.location = self.np_random.uniform(low=self.radius, high=(self.x_threshold - self.radius), size=(2,))
        self.velocity = self.np_random.uniform(low=-10, high=10, size=(2,))
        self.state = np.concatenate((self.location, self.velocity))

        self.target_points = self.np_random.uniform(low=self.radius, high=(self.x_threshold - self.radius), size=(2,))

        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 500
        screen_height = 500

        scale = screen_width / self.x_threshold

        x, y = self.state[0], self.state[1]

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            self.curling = rendering.make_circle(self.radius * scale)
            self.curling_transform = rendering.Transform(translation=(int(x * scale), int(y * scale)))
            self.curling.add_attr(self.curling_transform)
            self.viewer.add_geom(self.curling)
            self.viewer.draw_circle(3, color=(1, 0, 0)).add_attr(rendering.Transform(
                (int(self.target_points[0] * scale), int(self.target_points[1] * scale))))

        self.curling_transform.set_translation(int(x * scale), int(y * scale))

        self.viewer.draw_circle(3, color=(1, 0, 0)).add_attr(rendering.Transform(
            (int(self.target_points[0] * scale), int(self.target_points[1] * scale))))

        if self.state is None:
            return None

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _distance(self, x, y):
        target_x, target_y = self.target_points[0], self.target_points[1]

        distance = np.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)

        return distance

    def __step(self, x, y, x_dot, y_dot, x_force, y_force):
        next_x = x + x_dot * self.seconde_per_step + 0.5 * x_force * (self.seconde_per_step ** 2) / self.mass
        next_y = y + y_dot * self.seconde_per_step + 0.5 * y_force * (self.seconde_per_step ** 2) / self.mass

        if next_x + self.radius >= self.x_threshold:
            next_x = 2 * self.x_threshold - next_x - 2 * self.radius
            x_dot = -0.9 * (x_dot + x_force * self.seconde_per_step / self.mass)
            y_dot = 0.9 * (y_dot + y_force * self.seconde_per_step / self.mass)
        elif next_x - self.radius <= 0:
            next_x = 2 * self.radius - next_x
            x_dot = -0.9 * (x_dot + x_force * self.seconde_per_step / self.mass)
            y_dot = 0.9 * (y_dot + y_force * self.seconde_per_step / self.mass)
        else:
            next_x = next_x
            x_dot = x_dot + x_force * self.seconde_per_step / self.mass

        if next_y + self.radius >= self.x_threshold:
            next_y = 2 * self.x_threshold - next_y - 2 * self.radius
            x_dot = 0.9 * (x_dot + x_force * self.seconde_per_step / self.mass)
            y_dot = -0.9 * (y_dot + y_force * self.seconde_per_step / self.mass)
        elif next_y - self.radius <= 0:
            next_y = 2 * self.radius - next_y
            x_dot = 0.9 * (x_dot + x_force * self.seconde_per_step / self.mass)
            y_dot = -0.9 * (y_dot + y_force * self.seconde_per_step / self.mass)
        else:
            next_y = next_y
            y_dot = y_dot + y_force * self.seconde_per_step / self.mass

        return next_x, next_y, x_dot, y_dot
