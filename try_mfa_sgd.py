import os
import torch
from torchvision.datasets import ImageFolder
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from mfa import MFA
from utils import ReshapeTransform, samples_to_mosaic
from imageio import imwrite

dataset = 'mnist'

print('Training for', dataset)
model_dir = './models/' + dataset

if dataset == 'mnist':
    trans = transforms.Compose([transforms.ToTensor(), ReshapeTransform([-1])])
    train_set = MNIST(root='./data', train=True, transform=trans, download=True)
    n_components = 50
    n_factors = 6
    n_features=28*28
else:
    w = 64
    trans = transforms.Compose([transforms.Resize(w), transforms.ToTensor(), ReshapeTransform([-1])])
    train_set = ImageFolder(root='/mnt/local/eitanrich/PhD/Datasets/CelebA/cropped', transform=trans)
    n_components = 100
    n_factors = 10
    n_features=w*w*3

model = MFA(n_components=n_components, n_features=n_features, n_factors=n_factors)
model.cuda()

print('MPPCA EM Training...')
ll_log_em = model.batch_fit(train_set, max_iterations=10, test_size=256, responsibility_sampling=0.2)
print('Saving model...')
torch.save(model.state_dict(), os.path.join(model_dir, 'model_mppca_em_c_{}_l_{}.pth'.format(n_components, n_factors)))
# model.load_state_dict(torch.load(os.path.join(model_dir, 'model_mppca_em_c_{}_l_{}.pth'.format(n_components, n_factors))))

print('Continuing with MFA SGD Training...')
ll_log_sgd = model.sgd_train(train_set, test_size=256, max_epochs=10, responsibility_sampling=0.25)
print('Saving model...')
torch.save(model.state_dict(), os.path.join(model_dir, 'model_mfa_sgd_c_{}_l_{}.pth'.format(n_components, n_factors)))
print('Done')

plt.plot(ll_log_em + ll_log_sgd)
plt.grid(True)
plt.show()
