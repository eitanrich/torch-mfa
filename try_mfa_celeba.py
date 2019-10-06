import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import RandomSampler
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
from mfa import MFA
import pickle as pkl

n_components = 100
trans = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
train_set = ImageFolder(root='/mnt/local/eitanrich/PhD/Datasets/CelebA/cropped', transform=trans)
print('Reading...')
samples, labels = zip(*[train_set[id] for id in RandomSampler(train_set, replacement=True, num_samples=20000)])
samples = torch.stack(samples)
samples = samples.reshape(-1, 64*64*3)
print(samples.shape, len(labels))
#
mfa = MFA(n_components=n_components, n_features=64*64*3, n_factors=10)
mfa.cuda()
print('Fitting using EM...')
mfa.fit(samples.cuda(), max_iterations=50)

print('Visualizing...')
n = 200
rnd_samples, c_nums = mfa.sample(n, with_noise=False)
rnd_samples = rnd_samples.cpu().numpy()

# pkl.dump((rnd_samples, c_nums), open('samples.pkl', 'wb'))
# rnd_samples, c_nums = pkl.load(open('samples.pkl', 'rb'))

rnd_samples = np.maximum(-0.2, np.minimum(1.2, rnd_samples))
all_samples = []
for c in range(n_components):
    all_samples.append([rnd_samples[i].reshape([3, 64, 64]).transpose([1, 2, 0]) for i in range(n) if c_nums[i] == c])

min_per_class = min([len(s) for s in all_samples])
mosaic = np.vstack([np.hstack([all_samples[c][i] for i in range(min_per_class)]) for c in range(n_components)])
plt.imshow(mosaic)
plt.axis('off')
plt.show()
