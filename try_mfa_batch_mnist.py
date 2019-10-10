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
# mfa.cuda()

print('Fitting using EM...')
model.batch_fit(train_set)
exit()

# print('Visualizing...')
# n = 1000
# rnd_samples, c_nums = mfa.sample(n, with_noise=False)
# rnd_samples = rnd_samples.cpu().numpy()
# rnd_samples = np.maximum(-0.2, np.minimum(1.2, rnd_samples))
# all_samples = []
# for c in range(n_components):
#     all_samples.append([rnd_samples[i].reshape([28, 28]) for i in range(n) if c_nums[i] == c])
#
# min_per_class = min([len(s) for s in all_samples])
# mosaic = np.vstack([np.hstack([all_samples[c][i] for i in range(min_per_class)]) for c in range(n_components)])
# plt.imshow(mosaic)
# plt.axis('off')
# plt.show()
