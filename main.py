from utils.Environment import DynamicMatching
from utils.Config import CONFIG
from utils.Agents import GreedyAgent, MaxQueueAgent, PrimalDualBlindAgent, PrimalDualAgent
from utils.Experiments import Experiment

if __name__ == "__main__":
    ''' Generate Env Config '''
    DM_config = CONFIG()

    ''' Build Env '''
    DM_ENV = DynamicMatching(DM_config)
    
    ''' Choose Agent for the Env '''
    # DM_AGENT = GreedyAgent()
    # DM_AGENT = MaxQueueAgent()
    # DM_AGENT = PrimalDualBlindAgent()
    DM_AGENT = PrimalDualAgent()
    
    ''' Settup & Run the Exp '''
    Greedy_Exp = Experiment(env = DM_ENV, agent = DM_AGENT)
    Greedy_Exp.run()