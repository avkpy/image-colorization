import numpy as np
import torch
from torchvision import transforms, models
import cv2
import PIL
import os
from sklearn.preprocessing import OneHotEncoder

def data_grayscale_sharder(npy_file, batch_size=20):
    data = np.load(open(npy_file, 'rb'))
    for i in range(0,data.shape[0],batch_size):
        np.save('shards/l/grayscale_'+(i/batch_size).__str__(), data[i:i+batch_size])

def data_ab_sharder(npy_files, batch_size=20):
    data = np.vstack([np.load(open(f, 'rb')) for f in npy_files])
    for i in range(0,data.shape[0],batch_size):
        np.save('shards/ab/ab_'+(i/batch_size).__str__(), data[i:i+batch_size])

def load_grayscale_data(npy_file):
    return np.load(open(npy_file, 'rb'))

def load_ab_data(npy_file):
    return np.load(open(npy_file, 'rb'))

def convert2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_Lab2RGB)

def convert2gray(img):
    return np.expand_dims(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), axis=0)

def convertlab2gray(img):
    return np.expand_dims(cv2.cvtColor(img, cv2.COLOR_Lab2GRAY), axis=0)

def get_normalize1():
    return transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485], std=[0.229])])

def get_normalize2():
    return transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456], std=[0.229, 0.224])])

def get_normalize3():
    return transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def preprocess_input_3(x):
    return get_normalize2()(PIL.Image.fromarray(x.astype(np.uint8)))

def preprocess_input_1(x):
    return get_normalize1()(PIL.Image.fromarray(x.astype(np.uint8)))

def target_dataloader(file_number):
    data = load_ab_data(os.path.join(os.path.dirname(__file__), 'shards_/ab/ab_'+file_number.__str__()+'.0.npy'))
    classes = 256
    one_hot_targets = []
    for d in data:
        targets = cv2.resize(d, (56,56)).reshape(56*56*2)
        one_hot_targets.append(np.expand_dims(np.eye(classes)[targets].reshape(56,56,2,256), axis=0)) # 20, 56*56*2, 256
    data = np.concatenate(one_hot_targets, axis=0)
    return torch.from_numpy(data)

def input_dataloader(file_number):
    data = load_grayscale_data(os.path.join(os.path.dirname(__file__), 'shards_/l/grayscale_'+file_number.__str__()+'.0.npy'))
    return torch.from_numpy(data).unsqueeze(3)
