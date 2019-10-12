import os
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
from mfa import MFA
from utils import ReshapeTransform, samples_to_mosaic
from imageio import imwrite

n_components = 50
trans = transforms.Compose([transforms.ToTensor(), ReshapeTransform([-1])])     # transforms.Normalize((0.5,), (0.5,))
train_set = MNIST(root='./data', train=True, transform=trans, download=True)

model = MFA(n_components=n_components, n_features=28*28, n_factors=6)
model.cuda()

print('Fitting using EM...')
model.batch_fit(train_set, max_iterations=15, responsibility_threshold=1e-3)

print('Saving model...')
model_dir = './models/mnist'
os.makedirs(model_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))

print('Generating new samples...')
rnd_samples, _ = model.sample(400, with_noise=False)
mosaic = samples_to_mosaic(rnd_samples.cpu().numpy(), image_shape=[28, 28])
imwrite(os.path.join(model_dir, 'samples.jpg'), mosaic)
