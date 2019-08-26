import json
import nltk
import os
import random
import re
import torch
import numpy as np
from tqdm import tqdm
import gc
from time import time
import tensorflow as tf
import tensorflow_probability as tfp
import factor_analysis
from sklearn.mixture import BayesianGaussianMixture
from torch.autograd import Variable
from torchvision import transforms, models
import PIL

from torch import nn, optim
import torch.nn.functional as F
import pandas as pd

import matplotlib.pyplot as plt
import cv2
import math
import threading

import model
from model import persistence

if __name__ == "__main__":

    print("Running model: press Ctrl+C to stop")
    
    model.run_model(1, 25000)