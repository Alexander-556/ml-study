"""Import Libraries and Custom Files"""

# Necessary Libraries
import numpy as np
import torch

#Normalizing the data helps us run torch on it more cleanly
def normalize(input:torch.Tensor)->torch.Tensor:
    avg_input = input.mean()
    std_input = input.std()
    norm_input = (input - avg_input) / std_input
    return norm_input