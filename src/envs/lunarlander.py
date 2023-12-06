__credits__ = ["Andrea PIERRÉ"]

import math
import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np

import gym
from gym import error, spaces

class LunarLander():

    def __init__(
        self, step_max
    ):
        self.gym_env = gym.make('LunarLander-v2')
        self.step_max = step_max
        self.steps = 0
        self.done = False
        self.success = False
        self.num_episodes = 0
        self.total_reward = 0

        #added parameter
        self._action_size = 4 #do nothing, fire left, orientation engine, fire main engine, fire right orientation engine
        self._state_size = 1
        self._n_state_variables = 8


        self.low = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                -1.5,
                -1.5,
                # velocity bounds is 5x rated speed
                -5.0,
                -5.0,
                -math.pi,
                -5.0,
                -0.0,
                -0.0,
            ]
        ).astype(np.float32)
        self.high = np.array(
            [
                # these are bounds for position
                # realistically the environment should have ended
                # long before we reach more than 50% outside
                1.5,
                1.5,
                # velocity bounds is 5x rated speed
                5.0,
                5.0,
                math.pi,
                5.0,
                1.0,
                1.0,
            ]
        ).astype(np.float32)

        # print(self.low)
        # print(self.high)

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(self.low, self.high)
        self.action_space = spaces.Discrete(4)

        # self._original_state_ranges = [(-1.5, 1.5),(-1.5, 1.5),(-5.0, 5.0),(-5.0, 5.0),(-math.pi, math.pi),(-5.0, 5.0),(-0.0, 1.0),(-0.0, 1.0)]

        self._original_state_ranges = [(-1.5, 1.5),(-1.5, 1.5),(-5.0, 5.0),(-5.0, 5.0),(-math.pi, math.pi),(-5.0, 5.0),(-0.0, 1.0),(-0.0, 1.0)]


        self._state_ranges = []
        for i in range (self._n_state_variables):
            # print(type(low[i]))
            low = math.floor(self._original_state_ranges[i][0]) 
            high = math.ceil(self._original_state_ranges[i][1]) + 1
            r = (low, high)
            self._state_ranges.append(r)
        
        
        # self._vars_split_allowed = [1 for i in range(len(self._state_ranges)-2)]

        self._vars_split_allowed = [1 for i in range(len(self._state_ranges) - 2)]



    def step(self, action):
        self.done = False
        # print(self.gym_env.step(action))
        next_state, reward, terminated, _, info = self.gym_env.step(action)
        # next_state, reward, terminated, info = self.gym_env.step(action)

        # terminates when collides or is not awake i.e. lands
        
        self.steps += 1
        if self.steps == self.step_max:
            self.done = True
        else:
            self.done = terminated

        if not self.gym_env.lander.awake:
            self.success = True
        else:
            self.success = False

        if self.done:
            # self.render()
            self.num_episodes += 1
        self.total_reward += reward

        info["done"] = self.done
        info["succ"] = self.success
        info["reward"] = self.total_reward
        info["steps"] = self.steps
        info["num_episodes"] = self.num_episodes
        # return np.array(state, dtype=np.float32), reward, terminated, False, {}
        # return next_state, reward, self.done, info
        # print(type(next_state))
        refined_next_state = next_state.tolist()
        for i in range(len(refined_next_state)):
            if refined_next_state[i] > self._original_state_ranges[i][1]:
                refined_next_state[i] = self._original_state_ranges[i][1]
            if refined_next_state[i] < self._original_state_ranges[i][0]:
                refined_next_state[i] = self._original_state_ranges[i][0]

        return  refined_next_state, reward, self.done, self.success
        # return  next_state.tolist(), reward, self.done, self.success


    def reset(self):
        self.steps = 0
        self.total_reward = 0
        self.done = False
        self.success = False 
        # print(self.gym_env.reset()[0].tolist())
        return self.gym_env.reset()[0].tolist()
        # return self.gym_env.reset()

    


    