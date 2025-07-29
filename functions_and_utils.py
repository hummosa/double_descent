import numpy as np
import matplotlib.pyplot as plt
from sklearn import logger
import torch
import os
import matplotlib.transforms as mtransforms
import pandas as pd
from collections import defaultdict
import plot_style
plot_style.set_plot_style()

def explore_data_container(data):
    """
    Explores a nested data container (list, numpy array, or PyTorch tensor).
    Args:
        data: The input data container.
    Returns:
        None
    """
    def print_info(layer, depth):
        if isinstance(layer, list):
            print(f"Layer {depth}: List, Length = {len(layer)}")
            for item in layer:
                print_info(item, depth + 1)
                break  # Only print the first item
        elif isinstance(layer, np.ndarray):
            print(f"Layer {depth}: Numpy Array, Shape = {layer.shape}")
        elif isinstance(layer, torch.Tensor):
            print(f"Layer {depth}: PyTorch Tensor, Shape = {tuple(layer.shape)}")
            return  # Stop exploring when a tensor is encountered
        elif isinstance(layer, dict):
            print(f"Layer {depth}: Dictionary, keys: {list(layer.keys())}")
            for key in layer.keys():
                print_info(layer[key], depth + 1)
        else:
            print(f"Layer {depth}: Unknown type")
    print_info(data, depth=0)

def stats(var, var_name=None):
    if type(var) == type([]): # if a list
        var = np.array(var)
    elif type(var) == type(np.array([])):
        pass #if already a numpy array, just keep going.
    else: #assume torch tensor
        pass
        # var = var.detach().cpu().numpy()
    if var_name:
        print(var_name, ':')   
    out = ('Mean, {:2.5f}, var {:2.5f}, min {:2.3f}, max {:2.3f}, norm {}'.format(var.mean(), var.var(), var.min(), var.max(),np.linalg.norm(var) ))
    print(out)
    return (out)
