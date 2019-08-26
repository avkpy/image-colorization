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
from torch.optim.lr_scheduler import MultiStepLR

import torch
from torch import nn, optim
import torch.nn.functional as F
import pandas as pd

import matplotlib.pyplot as plt
import cv2
import math
import threading

from . import vision
from .vision import *

from . import model
from .model import *

from . import preprocess
from .preprocess import *

from . import persistence
from .persistence import *

def thread_get_unknown(markers, i, input_gray, sess):
    with sess.graph.as_default():
        fg, unknown = get_unknown(get_threshold(input_gray[i].numpy()))
        marker = get_marker(fg, unknown)
        markers[i] = np.expand_dims(marker, axis=0).reshape(1,224,224,1)

def train(encoder, encoder_optimizer,
          criterion1,batch_size, max_length=25000, file_number=1, bgm=None):
    
    print_loss_total = 0
    print_every_file = 50

    losses = []

    print_loss_total = 0
    input_gray = input_dataloader(file_number)
    target_data = target_dataloader(file_number) # single target
    target_length = target_data.shape[0]
    markers = dict()
    
    with tf.compat.v1.Session() as sess:
        with sess.graph.as_default():
            try:
                threads = []
                for i in range(target_length):
                    t = threading.Thread(target=thread_get_unknown, args=(markers,i,input_gray,sess))
                    t.start()
                    threads.append(t)
                for i in range(target_length):
                    threads[i].join()
            except Exception as e:
                raise e
                
            markers = torch.from_numpy(np.concatenate([markers[i] for i in sorted(markers)], axis=0)).long()
            encoder_target, encoder_deviation, encoder_latent = encoder.forward(input_gray, markers)
    
    loss = criterion1(encoder_target.view(-1,256).flatten().double(), target_data.view(-1,256).flatten().double())
    loss.backward()

    losses.append(loss.item())

    return losses, loss

def run_model(diff_epoch, max_length=5000):
    
    torch.cuda.empty_cache()

    epochs = 100
    print_every = 1
    plot_every = 1
    learning_rate = 0.01
    clip = 5

    batch_size = 20
    file_batch_size = 20
    vocab_size = 784
    embed_size = 27*27
    hidden_size = 4
    encoder_linear_output_size = 576
    decoder_hidden_size = batch_size

    encoder = EncoderRNN(vocab_size, embed_size, hidden_size, batch_size, decoder_hidden_size)
    encoder_optimizer = optim.Adam(encoder.parameters())
    encoder_optimizer.zero_grad()

    # scheduler = MultiStepLR(encoder_optimizer, [3,5], gamma=0.9)

    torch.autograd.set_detect_anomaly(True)

    criterion1 = torch.nn.BCELoss()

    if os.path.isfile('img_color.checkpoint.pth'):
        start_epoch, loss, losses, f_number = load_model(encoder, encoder_optimizer)
        f_number=0
        print("file number is: ", f_number, " file number changed")
    else:
        start_epoch = 0
        f_number = 0
        losses = []
        print("Fresh start: ", time(), " seconds")
    
    encoder.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in tqdm(range(start_epoch+1, start_epoch+2)):
        # scheduler.step()
        # print("Epoch: ", epoch, "LR: ", scheduler.get_lr())
        for file_number in tqdm(range(f_number, int(max_length/file_batch_size))):
            
            gc.collect()

            l, loss = train(encoder, encoder_optimizer, criterion1, batch_size, max_length, file_number)
            losses.append(l)

            nn.utils.clip_grad_norm_(encoder.parameters(), clip)
            encoder_optimizer.step()

            encoder_optimizer.zero_grad()

            if ((file_number+1) % 15) == 0:
                save_model(encoder, encoder_optimizer, epoch, losses, loss, file_number)
            else:
                print("File loss: ", losses[-1])
        print("Epoch loss: " , losses[-1])
