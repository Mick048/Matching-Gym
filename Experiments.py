import numpy as np

class Experiment(object):
    """Optional instrumentation for running an experiment.

    Runs a simulation between an arbitrary openAI Gym environment and an algorithm, saving a dataset of (reward, time, space) complexity across each episode,
    and optionally saves trajectory information.

    Attributes:
        seed: random seed set to allow reproducibility
        dirPath: (string) location to store the data files
        nEps: (int) number of episodes for the simulation
        deBug: (bool) boolean, when set to true causes the algorithm to print information to the command line
        env: (openAI env) the environment to run the simulations on
        epLen: (int) the length of each episode
        numIters: (int) the number of iterations of (nEps, epLen) pairs to iterate over with the environment
        save_trajectory: (bool) boolean, when set to true saves the entire trajectory information
        render_flag: (bool) boolean, when set to true renders the simulations
        agent: (or_suite.agent.Agent) an algorithm to run the experiments with
        data: (np.array) an array saving the metrics along the sample paths (rewards, time, space)
        trajectory_data: (list) a list saving the trajectory information
    """

    def __init__(self, env, agent):
        '''
        Args:
            env: (openAI env) the environment to run the simulations on
            agent: (or_suite.agent.Agent) an algorithm to run the experiments with
            dict: a dictionary containing the arguments to send for the experiment, including:
                dirPath: (string) location to store the data files
                nEps: (int) number of episodes for the simulation
                deBug: (bool) boolean, when set to true causes the algorithm to print information to the command line
                env: (openAI env) the environment to run the simulations on
                epLen: (int) the length of each episode
                numIters: (int) the number of iterations of (nEps, epLen) pairs to iterate over with the environment
                save_trajectory: (bool) boolean, when set to true saves the entire trajectory information
                render: (bool) boolean, when set to true renders the simulations
                pickle: (bool) when set to true saves data to a pickle file
        '''

        self.seed = 12
        self.env = env
        self.epLen = 10
        self.num_iters = 1
        self.agent = agent

        np.random.seed(self.seed)  # sets seed for experiment

    # Runs the experiment
    def run(self):
        '''
            Runs the simulations between an environment and an algorithm
        '''
        for ite in range(self.num_iters):  # loops over the episodes

            # Reset the environment
            self.env.reset()

            # Reset the agent
            self.agent.reset()
            self.agent.update_config(self.env, self.env.get_config())


            oldState = self.env.initial_queue # obtains old state
            arrival = self.env.initial_queue
            epReward = 0


            # repeats until episode is finished
            for t in range(self.epLen):
                print("\n"+"="*10 + f"{t}-th period" + "="*10)
                # Select action list
                action_list = self.agent.pick_action(
                        queue=oldState, arrival=arrival,t=t)
                print(f"Proposed Action List: {action_list}")

                # steps based on the action chosen by the algorithm
                queue, physical_rewards, virtual_rewards, episode_over, new_arrival, remain_action_list = self.env.step(action_list)
                print(f"Remain Action List: {remain_action_list}")

                epReward += physical_rewards

                oldState = queue[self.agent.virtual]
                arrival = new_arrival

                # Update the Policy
                self.agent.update_policy(arrival=arrival, remain_action_list=remain_action_list, t=t)

            print(f"\nTotal Rewards: {epReward}")

            self.env.close()
