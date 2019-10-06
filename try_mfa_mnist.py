import torch
from torchvision.datasets import MNIST
from torch.utils.data import RandomSampler
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
from mfa import MFA
import pickle as pkl

n_components = 30
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = MNIST(root='./data', train=True, transform=trans, download=True)
print('Reading...')
samples, labels = zip(*[train_set[id] for id in RandomSampler(train_set, replacement=True, num_samples=30000)])
samples = torch.stack(samples)
samples = samples.reshape(-1, 28*28)
print(samples.shape, len(labels))

mfa = MFA(n_components=n_components, n_features=28*28, n_factors=6)
mfa.fit(samples, max_iterations=20)

n = 1000
rnd_samples, c_nums = mfa.sample(n, with_noise=False)
rnd_samples = rnd_samples.numpy()
pkl.dump((rnd_samples, c_nums), open('samples.pkl', 'wb'))

rnd_samples, c_nums = pkl.load(open('samples.pkl', 'rb'))
rnd_samples = np.maximum(-0.2, np.minimum(1.2, rnd_samples))
all_samples = []
for c in range(n_components):
    all_samples.append([rnd_samples[i].reshape([28, 28]) for i in range(n) if c_nums[i] == c])

min_per_class = min([len(s) for s in all_samples])
mosaic = np.vstack([np.hstack([all_samples[c][i] for i in range(min_per_class)]) for c in range(n_components)])
plt.imshow(mosaic)
plt.axis('off')
plt.show()
