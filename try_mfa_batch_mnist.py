import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
from mfa import MFA
from utils import ReshapeTransform

n_components = 50
trans = transforms.Compose([transforms.ToTensor(), ReshapeTransform([-1])])     # transforms.Normalize((0.5,), (0.5,))
train_set = MNIST(root='./data', train=True, transform=trans, download=True)

model = MFA(n_components=n_components, n_features=28*28, n_factors=6)
model.cuda()

print('Fitting using EM...')
model.batch_fit(train_set)

