import numpy as np


"""
The config of the environment of Dynamic Matching problem is defined as follows:

Config = {
    T: int, # number of time periods
    AR: np.ndarry, # Arrival Rate -> shape: (T, d)
    M: np.ndarry, # Mathces -> shape: (m, d)
    R: np.ndarry, # Reward -> shape: (T, m)
    IQ: np.ndarry, # Initial Queue -> shape: (d)
    r: float, # Discount Rate: float
    a: float  # Abandon Rate
}

where T is the number of time periods, m is the number of match types, d is the number of agent types.
"""

matches = np.array([
    [1,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,0,1,0,0,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,1],
    [1,0,0,0,0,1,0,0],
    [1,0,0,0,0,0,1,0],
    [1,0,0,0,0,0,0,1],
    [0,1,0,0,0,0,0,1],
    [0,0,1,0,1,0,0,0],
    [0,0,1,0,0,1,0,0],
    [0,0,1,0,0,0,0,1],
    [0,0,0,1,1,0,0,0],
])

rewards = [0., 0., 0., 0., 0., 0.,
           0., 0., 0.99530526, 1.0054256,
           0.99536582, 0.9953427, 1.00241962,
           0.9808672, 0.98275082, 0.99437712]

arrival_rates = np.array([
    0.12493259, 0.12414321, 0.12512027, 0.12620845,
    0.12402401, 0.12402403, 0.1262783, 0.12526913
])

arrival = np.array([0, 0, 0, 0, 0, 0, 1, 0])


DM_config ={
    "T": 10,
    "AR": np.tile(arrival_rates, (10, 1)),
    "M": matches,
    "R": np.tile(rewards, (10, 1)),
    'IQ': arrival,
    "r": 1.,
    "a": 0.
}

def CONFIG():
    return DM_config

