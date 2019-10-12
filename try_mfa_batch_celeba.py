import os
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from mfa import MFA
from utils import ReshapeTransform, samples_to_mosaic
from imageio import imwrite

model_dir = './models/celeba'
n_components = 200
trans = transforms.Compose([transforms.Resize(64), transforms.ToTensor(), ReshapeTransform([-1])])
train_set = ImageFolder(root='/mnt/local/eitanrich/PhD/Datasets/CelebA/cropped', transform=trans)

model = MFA(n_components=n_components, n_features=64*64*3, n_factors=10)
model.cuda()

model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')))

# print('Fitting using EM...')
# model.batch_fit(train_set, max_iterations=10, responsibility_sampling=0.2, responsibility_threshold=1e-3)

# print('Saving model...')
# os.makedirs(model_dir, exist_ok=True)
# torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))

print('Generating new samples...')
rnd_samples, _ = model.sample(12*12, with_noise=False)
rnd_samples = rnd_samples.cpu().numpy().reshape(-1, 3, 64, 64).transpose([0, 2, 3, 1]).reshape(-1, 64*64*3)
mosaic = samples_to_mosaic(rnd_samples, image_shape=[64, 64, 3])
imwrite(os.path.join(model_dir, 'samples.jpg'), mosaic)
