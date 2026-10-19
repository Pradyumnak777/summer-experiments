'''
this is to evaluate if the trained models can perform semantic disnentaglement-


1. assume intensity of the gene burst is a factor we want to manipulate..
'''

import torch
from torch.utils.data import DataLoader
from VAE_disent.model import twoChannelVAE
from VAE_disent.data_utils import twoChannelDataset

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import debugpy
# debugpy.listen(("127.0.0.1", 5678)) #127.0.0.1, cz only local host can talk to the port
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()
# print("Debugger attached! Running code...")