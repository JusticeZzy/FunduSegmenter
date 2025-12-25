"""
MIT License

Copyright (c) 2025 Zhenyi Zhao
Copyright (c) University of Dundee
"""


import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)  
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)