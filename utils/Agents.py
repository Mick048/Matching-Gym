import numpy as np
from utils.Static_Solver import StaticSolver

'''
All agents should inherit from the Agent class.
'''
class Agent(object):

    def __init__(self):
        pass
    def reset(self):
        pass
    def update_config(self, env, config):
        ''' Update agent information according to config file'''
        pass

    def update_policy(self, arrival=None, remain_action_list=None, t=None):
        '''Update policy based record history'''
        pass
    def pick_action(self, obs, h):
        '''Select an action based on the observation'''
        pass


""" Greedy Agent """
class GreedyAgent(Agent):
    def __init__(self):
        self.matches = None
        self.m = None
        self.rewards = None
        self.arrivals = None
        self.virtual = False

    def reset(self):
        self.matches = None
        self.m = None
        self.rewards = None
        self.arrivals = None
        self.virtual = False

    def update_config(self, env, config):
        ''' Update agent information based on the config__file'''
        self.matches = config["M"]
        self.m = self.matches.shape[0]
        self.rewards = config["R"]
        self.arrivals = config["IQ"] # Inital Queue
    
    def update_policy(self, arrival=None, remain_action_list=None, t=None):
        '''Update policy based record history'''
        pass

    def pick_action(self, queue, arrival, t):
        '''Select an action based upon the observation'''
        best_reward = 0
        best_match_idx = None

        for i in range(self.m):
            # Check if match uses current arrival
            if np.inner(self.matches[i], arrival) > 1e-5:
                # Check if we have enough agents
                if np.all(queue >= self.matches[i]) and self.rewards[t][i] > best_reward:
                    best_match_idx = i
                    best_reward = self.rewards[t][i]
        return [best_match_idx]

""" MaxQueue Agent """
class MaxQueueAgent(Agent):
    def __init__(self):
        self.matches = None
        self.m = None
        self.rewards = None
        self.arrivals = None
        self.arrival_rates = None
        self.virtual = False
        self.valid_indices = None

    def reset(self):
        self.matches = None
        self.m = None
        self.rewards = None
        self.arrivals = None
        self.arrival_rates = None
        self.virtual = False
        self.valid_indices = None

    def update_config(self, env, config):
        ''' Update agent information based on the config__file'''
        self.matches = config["M"]
        self.m = self.matches.shape[0]
        self.rewards = config["R"]
        self.arrivals = config["IQ"] # Inital Queue
        self.arrival_rates = config["AR"] # arrival rates

        # Get optimal offline solution
        solver = StaticSolver(self.matches, self.arrival_rates[0], self.rewards[0])
        solver.solve()  # Will use gurobi if available
        primal_soln = solver.get_primal_solution()

        # Get valid matches (positive in optimal solution)
        self.valid_indices = np.where(primal_soln > 1e-10)[0]
        print(f"Valid indices: {self.valid_indices}")

        # Validate solution
        if len(self.valid_indices) != self.matches.shape[1]:
            raise ValueError("Primal solution not basic feasible")

    def update_policy(self, arrival=None, remain_action_list=None, t=None):
        '''Update policy based record history'''
        pass

    def pick_action(self, queue, arrival, t):
        '''Select an action based upon the observation'''
        best_match_idx = None
        highest_sum = 0

        for i in self.valid_indices:
            if np.all(queue >= self.matches[i]):
                queue_sum = np.inner(self.matches[i], queue)
                if queue_sum > highest_sum:
                    best_match_idx = i
                    highest_sum = queue_sum

        return [best_match_idx]

""" Primal-Dual Blind Agent """
class PrimalDualBlindAgent(Agent):
    def __init__(self):
        self.T = None
        self.matches = None
        self.m = None
        self.rewards = None
        self.arrivals = None
        self.arrival_rates = None
        self.arrival_sum = None
        self.virtual = True # Using Virtual Queue as State
        self.valid_indices = None
        self.unrealized_matches = []
        self.v_vec = None
        self.solver = None

    def reset(self):
        self.T = None
        self.matches = None
        self.m = None
        self.rewards = None
        self.arrivals = None
        self.arrival_rates = None
        self.arrival_sum = None
        self.virtual = True # Using Virtual Queue as State
        self.valid_indices = None
        self.unrealized_matches = []
        self.v_vec = None
        self.solver = None

    def update_config(self, env, config):
        ''' Update agent information based on the config__file'''
        self.T = config["T"]
        self.matches = config["M"]
        self.m = self.matches.shape[0]
        self.rewards = config["R"]
        self.arrivals = config["IQ"] # Inital Queue
        self.arrival_rates = config["AR"] # arrival rates
        self.arrival_sum = config["IQ"]
        self.v_vec = [100.0] * self.T

        # Initialize solver with empirical arrival rates
        self.solver = StaticSolver(self.matches, self.arrival_sum, self.rewards[0])


    def update_policy(self, arrival=None, remain_action_list=None, t=None):
        '''Update internal policy based upon records'''

        # Update empirical arrival rates with a minimum threshold to prevent numerical issues
        self.arrival_sum = self.arrival_sum + arrival
        empirical_rates = self.arrival_sum / (t + 2)
        self.solver.update_arrival_rates(empirical_rates)

        self.unrealized_matches = remain_action_list


    def pick_action(self, queue, arrival, t):
        '''Select an action based upon the observation'''


        # Get empirical dual values
        self.solver.solve()
        dual_values = self.solver.get_dual_solution()

        # Find best match based on reduced reward
        best_match_idx = None
        highest_reduced_reward = 1e-6

        for m in range(self.m):
            # Only consider matches using current arrival
            if np.inner(self.matches[m], arrival) > 0:
                # Compute reduced reward with queue pressure
                reduced_reward = (
                    self.rewards[t][m] -
                    np.inner(dual_values, self.matches[m]) +
                    np.inner(queue, self.matches[m]) / self.v_vec[t]
                )
                if reduced_reward > highest_reduced_reward:
                    best_match_idx = m
                    highest_reduced_reward = reduced_reward

        self.unrealized_matches.append(best_match_idx)

        return self.unrealized_matches

""" PrimalDualAgent """
class PrimalDualAgent(Agent):
    def __init__(self):
        self.T = None
        self.matches = None
        self.m = None
        self.rewards = None
        self.arrivals = None
        self.arrival_rates = None
        self.virtual = True # Using Virtual Queue as State
        self.unrealized_matches = []
        self.v_vec = None
        self.solver = None

    def reset(self):
        self.T = None
        self.matches = None
        self.m = None
        self.rewards = None
        self.arrivals = None
        self.arrival_rates = None
        self.virtual = True # Using Virtual Queue as State
        self.unrealized_matches = []
        self.v_vec = None
        self.solver = None

    def update_config(self, env, config):
        ''' Update agent information based on the config__file'''
        self.T = config["T"]
        self.matches = config["M"]
        self.m = self.matches.shape[0]
        self.rewards = config["R"]
        self.arrivals = config["IQ"] # Inital Queue
        self.arrival_rates = config["AR"] # arrival rates
        self.arrival_sum = config["IQ"]
        self.v_vec = [100.0] * self.T

        # Initialize solver with empirical arrival rates
        self.solver = StaticSolver(self.matches, self.arrival_rates[0], self.rewards[0])



    def update_parameters(self, param):
        pass

    def update_obs(self, obs, action, reward, newObs, timestep, info):
        '''Add observation to records'''
        pass

    def update_policy(self, arrival=None, remain_action_list=None, t=None):
        '''Update internal policy based upon records'''
        self.unrealized_matches = remain_action_list


    def pick_action(self, queue, arrival, t):
        '''Select an action based upon the observation'''


        # Get empirical dual values
        self.solver.solve()
        dual_values = self.solver.get_dual_solution()

        # Find best match based on reduced reward
        best_match_idx = None
        highest_reduced_reward = 1e-6

        for m in range(self.m):
            # Only consider matches using current arrival
            if np.inner(self.matches[m], arrival) > 0:
                # Compute reduced reward with queue pressure
                reduced_reward = (
                    self.rewards[t][m] -
                    np.inner(dual_values, self.matches[m]) +
                    np.inner(queue, self.matches[m]) / self.v_vec[t]
                )
                if reduced_reward > highest_reduced_reward:
                    best_match_idx = m
                    highest_reduced_reward = reduced_reward

        self.unrealized_matches.append(best_match_idx)

        return self.unrealized_matches

