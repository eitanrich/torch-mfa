from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
from mfa import MFA
from utils import ReshapeTransform

n_components = 200
trans = transforms.Compose([transforms.Resize(64), transforms.ToTensor(), ReshapeTransform([-1])])
train_set = ImageFolder(root='/mnt/local/eitanrich/PhD/Datasets/CelebA/cropped', transform=trans)

model = MFA(n_components=n_components, n_features=64*64*3, n_factors=10)
model.cuda()

print('Fitting using EM...')
model.batch_fit(train_set, max_iterations=10, responsibility_sampling=0.2, responsibility_threshold=0.01)
