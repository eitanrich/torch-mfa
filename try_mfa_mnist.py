import os
import torch
from torchvision.datasets import MNIST
from torch.utils.data import RandomSampler
import torchvision.transforms as transforms
from mfa import MFA
from utils import ReshapeTransform, samples_to_mosaic
from imageio import imwrite

trans = transforms.Compose([transforms.ToTensor(), ReshapeTransform([-1])])
train_set = MNIST(root='./data', train=True, transform=trans, download=True)

print('Reading...')
samples, labels = zip(*[train_set[id] for id in RandomSampler(train_set)])
samples = torch.stack(samples)

model = MFA(n_components=50, n_features=28*28, n_factors=6)
model.cuda()

print('Fitting using EM...')
model.fit(samples.cuda(), max_iterations=20)

print('Generating new samples...')
model_dir = './models/mnist'
os.makedirs(model_dir, exist_ok=True)
rnd_samples, _ = model.sample(400, with_noise=False)
mosaic = samples_to_mosaic(rnd_samples.cpu().numpy(), image_shape=[28, 28])
imwrite(os.path.join(model_dir, 'samples_full_data_EM.jpg'), mosaic)
