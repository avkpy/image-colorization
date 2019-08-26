from . import vision
from .vision import *
from .preprocess import *
import numpy as np
import tensorflow as tf
import torch
from torch.autograd import Variable
from torchvision import transforms, models
import factor_analysis
from torch import nn, optim
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import math
import threading
from sklearn.mixture import BayesianGaussianMixture
from torchvision.models import inception_v3

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, batch_size, decoder_hidden_size,
                 lstm_layers=1, filter_size=2, super_filter_size=4):
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.filter_size = filter_size
        self.super_filter_size = super_filter_size
        self.batch_size = batch_size

        self.model_inceptionv3 = inception_v3(pretrained=True)

        for i, param in self.model_inceptionv3.named_parameters():
            param.requires_grad = False

        num_ftrs = self.model_inceptionv3.fc.in_features
        self.model_inceptionv3.fc = nn.Linear(num_ftrs, num_ftrs*128)

        ct = []
        for name, child in self.model_inceptionv3.named_children():
            for params in child.parameters():
                params.requires_grad = False
        
        # encoder - input
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

        self.batchnorm1 = nn.BatchNorm2d(512)
        
        self.sequential_input_encoder = nn.Sequential(
            self.conv1,
            self.conv2,
            nn.ReLU(),
            self.conv3,
            self.conv4,
            self.conv5,
            nn.ReLU(),

        )

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, output, markers):
        # 8 x 56 x 56
        output = output.view(self.batch_size,1,224,224).float()
        markers = markers.view(self.batch_size,1,224,224).float()

        conv_output = self.sequential_input_encoder(output)
        markers = self.sequential_input_encoder(markers)
        
        o = torch.zeros((self.batch_size,1,299,299))
        for i in range(self.batch_size):
            o[i,0,:,:] = torch.from_numpy(cv2.resize(output[i,0,:,:].detach().numpy(), (299,299)))
        inception_output, _ = self.model_inceptionv3(torch.cat([o.view(self.batch_size,1,299,299)]*3, dim=1)) # 20, 32768
        inception_output, _ = torch.sort(inception_output, descending=True)
        inception_output = inception_output.reshape(self.batch_size,512,512) # unknown is weighted mean, known is element-wise
        conv_output = conv_output.view(self.batch_size,512,-1)
        fused_inception = torch.bmm(inception_output, conv_output)
        output = conv_output.view(self.batch_size,512,-1) * fused_inception.view(self.batch_size,inception_output.shape[1],conv_output.shape[2]) # bias
        output = output.view(self.batch_size,512,56,56)
        markers = markers.view(self.batch_size,512,56,56)
        output = self.batchnorm1(output)
        markers = self.batchnorm1(markers)
        target = self.sigmoid(output + markers).view(self.batch_size,56,56,2,256)

        return target, markers, output

class FactorAnalysis():
    
    def __init__(self, data, covariance_prior, means, dimensions=2):
        self.f = factor_analysis.factors.Factor(data, factor_analysis.posterior.Posterior(covariance_prior, means))
        self.noise = factor_analysis.noise.Noise(self.f, self.f.posterior)
        self.set_variables()
    
    def set_variables(self, dimensions=2):
        self.factor = self.f.create_factor()
        self._noise = self.noise.create_noise(self.factor)
        # with tf.Session() as sess:
        #     self.vars = dict(zip(["Lambda", "Noise"], [
        #         Variable(torch.from_numpy(self.factor.eval()) * 1e10, requires_grad=False),
        #         Variable(torch.from_numpy(self.noise.noise.eval()), requires_grad=False)
        #     ]))
    
    
    @staticmethod
    def _data(marker, unknown):
        pixel_value = np.unique(marker)
        light_intensity = np.array([])
        pv = np.array([])
        for p in pixel_value:
            u = unknown[np.where(marker == p)]
            num_values = len(u)
            light_intensity = np.append(light_intensity, u.flatten())
            pv = np.append(pv, [p]*num_values)
        return np.concatenate([np.expand_dims(light_intensity, axis=1), np.expand_dims(pv, axis=1)], axis=1)
    @staticmethod
    def _mean(data): # aggregated data
        gm = BayesianGaussianMixture(n_components=2)
        gm.fit(data)
        return gm.means_
    @staticmethod
    def _covariance(data): # coming from distribution
        return np.cov(data.T)
    @staticmethod
    def _prob(uvdata, output, factor, noise): # x, z
        uvdata = uvdata.reshape(224*224,2)
        mean = np.mean(uvdata, axis=0)
        eps = (uvdata - mean).reshape(2,224*224).astype(np.float64) \
        - np.matmul(factor, output.reshape(2,224*224).astype(np.float64))
        prob = 1/(2*np.pi) * (np.exp(-0.5*np.matmul(np.transpose(eps), 
             np.linalg.pinv(np.diag(noise).reshape(1,-1))) * np.transpose(eps))).reshape(224*224,2)
        # normalizing the probability
        return (prob / np.sum(prob, axis=0)).reshape(224,224,2)

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor