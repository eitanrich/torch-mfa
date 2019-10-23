import os
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import RandomSampler
import torchvision.transforms as transforms
from mfa import MFA
from utils import ReshapeTransform, samples_to_mosaic
from imageio import imwrite

w = 64
n_components = 200
trans = transforms.Compose([transforms.Resize(w), transforms.ToTensor(), ReshapeTransform([-1])])
train_set = ImageFolder(root='/mnt/local/eitanrich/PhD/Datasets/CelebA/cropped', transform=trans)

print('Reading...')
samples, labels = zip(*[train_set[id] for id in RandomSampler(train_set, replacement=True, num_samples=50000)])
samples = torch.stack(samples)
samples = samples.reshape(-1, 64*64*3)

model = MFA(n_components=n_components, n_features=64*64*3, n_factors=10)
model.cuda()
print('Fitting using EM...')
model.fit(samples.cuda(), max_iterations=20, responsibility_sampling=0.2)

print('Generating new samples...')
model_dir = './models/celeba'
os.makedirs(model_dir, exist_ok=True)
rnd_samples, _ = model.sample(100, with_noise=False)
rnd_samples = rnd_samples.cpu().numpy().reshape([-1, 3, w, w]).transpose([0, 2, 3, 1]).reshape([-1, w*w*3])
mosaic = samples_to_mosaic(rnd_samples, image_shape=[w, w, 3])
imwrite(os.path.join(model_dir, 'samples_full_data_EM.jpg'), mosaic)
