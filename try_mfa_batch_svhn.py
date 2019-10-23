import os
import torch
from torchvision.datasets import SVHN
import torchvision.transforms as transforms
from mfa import MFA
from utils import ReshapeTransform, samples_to_mosaic
from matplotlib import pyplot as plt
from imageio import imwrite

w = 32
model_dir = './models/svhn'
os.makedirs(model_dir, exist_ok=True)
trans = transforms.Compose([transforms.ToTensor(), ReshapeTransform([-1])])     #
train_set = SVHN(root='../../Datasets/SVHN', transform=trans, split='extra', download=True)

for n_components in [500]:  # [50, 200, 500]:
    for n_factors in [8]:  # [2, 10, 20]:
        for batch_size in [5000]:   #[1000, 10000]:
            for try_num in range(1):
                model = MFA(n_components=n_components, n_features=w*w*3, n_factors=n_factors)
                model.cuda()

                print('EM fitting: {} components / {} factors / batch size {} / try {}...'.format(
                    n_components, n_factors, batch_size, try_num))

                ll_log = model.batch_fit(train_set, batch_size=batch_size, max_iterations=12, responsibility_sampling=0.2)

                plt.plot(ll_log, label='c{}_l{}_b{}'.format(n_components, n_factors, batch_size))
                plt.grid(True)
                plt.pause(0.1)
                rnd_samples, _ = model.sample(400, with_noise=False)
                rnd_samples = rnd_samples.cpu().numpy().reshape([-1, 3, w, w]).transpose([0, 2, 3, 1]).reshape([-1, w*w*3])
                mosaic = samples_to_mosaic(rnd_samples, image_shape=[w, w, 3])
                imwrite(os.path.join(model_dir, 'samples_c_{}_l_{}_b_{}_try_{}.jpg'.format(
                    n_components, n_factors, batch_size, try_num)), mosaic)

print('Done')
plt.show()

