import pickle
import os
from time import time
from . import model
from .model import *
import torch
from torch import nn, optim
from tqdm import tqdm

def save_model(encoder, optimizer, epoch, losses, loss, file_number):
    torch.save({
        'epoch': epoch,
        'file_number': file_number,
        'model_state_dict': encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'losses': losses
    }, 'img_color.checkpoint.pth')

def load_model(encoder, optimizer):
    checkpoint = torch.load('img_color.checkpoint.pth')
    encoder.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    losses = checkpoint['losses']
    file_number = checkpoint['file_number']

    return epoch, loss, losses, file_number