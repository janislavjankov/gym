"""
Cart-pole system which allows for noop, besides just pushing from the left or from the right.
"""

import math
from gym import spaces, logger
from gym.envs.classic_control.cartpole import CartPoleEnv
import numpy as np


class CartPoleTwoButtonsEnv(CartPoleEnv):
    """
    Description:
        Same as CartPole-v0

    Source:
        Same as CartPole-v0

    Observation:
        Same as CartPole-v0

    Actions:
        Type: MultiDiscrete([2, 2])
        Num	Action
        0,0	No-op (no button pushed)
        1,0 Push cart to the left
        0,1	Push cart to the right
        1,1 No-op (both buttons pushed)

    Reward:
        Same as CartPole-v0

    Starting State:
        Same as CartPole-v0

    Episode Termination:
        Same as CartPole-v0
    """

    def __init__(self):
        super().__init__()
        self.action_space = spaces.MultiDiscrete([2, 2])  # Can also be MultiBinary

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot = state

        # This is the only difference from CartPole-v0 -- accounts for the two buttons
        direction = action[1] - action[0]
        force = self.force_mag * direction

        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
        self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        self.state = (x, x_dot, theta, theta_dot)
        done = x < -self.x_threshold \
               or x > self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}
