from dataclasses import dataclass
from typing import Literal


@dataclass
class Config:
    d : int # reg dimension also parameters
    n : int # datapoints
    n_itr : int # number of iterations to run.
    lr : float = 0.01 # learning rate
    w_init : float = 0.0 # initial value of the parameters
    n_ins : int = 20 # number of models to run simultaneously
    noise_std : float = 0.1 # noise magnitude
    noise_type : Literal['input', 'output', 'time-correlated'] = 'input' # type of noise to add
    dropout : float = 0.0 # drop out features. 
    noise_std_time : float = 0.1 # standard deviation of time-correlated noise
