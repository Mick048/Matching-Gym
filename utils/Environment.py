import numpy as np
import gymnasium as gym


class DynamicMatching(gym.Env):
    """
    An environment representing the dynamic mathcing problem

    Attributes:
        T: int, # number of time periods
        AR: np.ndarry, # Arrival Rate -> shape: (T, d)
        M: np.ndarry, # Mathces -> shape: (m, d)
        R: np.ndarry, # Reward -> shape: (T, m)
        PQ: np.ndarry, # Pysical Queue -> shape: (d,)
        VQ: np.ndarry, # Virtual Queue -> shape: (d,)
        r: float, # Discount Rate: float
        a: float  # Abandon Rate

    where T is the number of time periods, m is the number of match types, d is the number of agent types.
    """
    def __init__(self, config):

        # Intializes Env parameters based on configuration dictionary
        self.config = config
        self.T = config["T"]
        self.AR = config["AR"]
        self.M = config["M"]
        self.R = config["R"]
        self.r = config["r"]
        self.a = config["a"]

        self.m = self.M.shape[0] # m is the number of match types
        self.d = self.M.shape[1] # d is the number of agent types
        self.seed = 16

        self.action_space = gym.spaces.Discrete(self.m, seed=self.seed)
        self.observation_space = gym.spaces.Discrete(self.d, seed=self.seed)
        self.initial_queue = config["IQ"].copy()
        self.physical_queue = self.initial_queue.copy()
        self.virtual_queue = self.initial_queue.copy()

        self.timestep = 0
        self.reset()

    def get_config(self):
        return self.config

    # Reset the environemnt to initial state
    def reset(self):
        self.timestep = 0
        self.physical_queue = self.initial_queue.copy()
        self.virtual_queue = self.initial_queue.copy()
        return

    # Defines one step of the DM, returning the new state, reward, whether time horizon is finished and unrealized action list
    def step(self, action_list, update=True):


        if self.timestep >= self.T:
            raise ValueError("Time horizon is finished") # Check the time horizon
        for i, action in enumerate(action_list):
            assert self.action_space.contains(action) or (action == None), f"Invalid action at index {i}: {action}" # Check the Action List is Valid

        # Update the Physical Queue
        # Realize the Action in the Action list.
        physical_rewards = 0
        actions_to_remove = []
        for action in action_list:
            if action == None:
                continue
            elif np.all(self.physical_queue >= self.M[action]):
                self.physical_queue = self.physical_queue - self.M[action]
                physical_rewards = physical_rewards + self.R[self.timestep, action]
                actions_to_remove.append(action)
        physical_rewards *= self.r ** (self.timestep) # Consider the discount rate at timestep t: reward * (r**t)

        # Remove the realized actions
        for action in actions_to_remove:
            action_list.remove(action)

        # Update the Virtual Queue
        # Do not need to realize the action list
        virtual_rewards = 0
        for action in action_list:
            if action == None:
                continue
            else:
                self.virtual_queue = self.virtual_queue - self.M[action]
                virtual_rewards = virtual_rewards + self.R[self.timestep, action]
        virtual_rewards *= self.r ** (self.timestep) # Consider the discount rate at timestep t: reward * (r**t)



        # Move to next period
        episode_over = False
        new_arrival = np.zeros(self.d)
        if update == True:
            self.timestep += 1
            if self.timestep == self.T:
                episode_over = True
            else:
                new_arrival_idx = self.observation_space.sample(probability=self.AR[self.timestep])
                new_arrival[int(new_arrival_idx)] = 1
                self.physical_queue = self.physical_queue + new_arrival
                self.virtual_queue = self.virtual_queue + new_arrival


        return [self.physical_queue.copy(), self.virtual_queue.copy()], physical_rewards, virtual_rewards, episode_over, new_arrival.copy(), action_list

