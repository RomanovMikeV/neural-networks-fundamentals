import torch
import numpy as np
import random

def seed_all(seed_value):
    ...
    ## YOUR CODE HERE
    # -- placeholder start --
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    # -- placeholder end --
