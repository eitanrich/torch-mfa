import os
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from mfa import MFA
from utils import ReshapeTransform, samples_to_mosaic
from matplotlib import pyplot as plt
from imageio import imwrite

w = 64
model_dir = './models/celeba'
# n_components = 200
trans = transforms.Compose([transforms.Resize(w), transforms.ToTensor(), ReshapeTransform([-1])])     #
train_set = ImageFolder(root='/mnt/local/eitanrich/PhD/Datasets/CelebA/cropped', transform=trans)

for n_components in [50, 200, 500]:
    for n_factors in [2, 10, 20]:
        for batch_size in [1000, 10000]:
            for try_num in range(1):
                model = MFA(n_components=n_components, n_features=w*w*3, n_factors=n_factors)
                model.cuda()

                print('EM fitting: {} components / {} factors / batch size {} / try {}...'.format(
                    n_components, n_factors, batch_size, try_num))

                ll_log = model.batch_fit(train_set, batch_size=batch_size, max_iterations=20)

                plt.plot(ll_log, label='c{}_l{}_b{}'.format(n_components, n_factors, batch_size))
                plt.grid(True)
                plt.pause(0.1)
                rnd_samples, _ = model.sample(400, with_noise=False)
                mosaic = samples_to_mosaic(rnd_samples.cpu().numpy(), image_shape=[28, 28])
                imwrite(os.path.join(model_dir, 'samples_c_{}_l_{}_b_{}_try_{}.jpg'.format(
                    n_components, n_factors, batch_size, try_num)), mosaic)

print('Done')
plt.show()



# model = MFA(n_components=n_components, n_features=w*w*3, n_factors=10)
# model.cuda()
#
# # model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth')))
#
#
# print('Fitting using EM...')
# # model.batch_fit(train_set, batch_size=1000, max_iterations=10, responsibility_sampling=0.2, responsibility_threshold=1e-3)
# ll_log = model.batch_fit(train_set, max_iterations=50, responsibility_sampling=0.2)
# plt.plot(ll_log)
# plt.grid(True)
# plt.savefig('celeba_mppca_minibatch_EM_init_with_lx2.pdf')
#
# print('Saving model...')
# os.makedirs(model_dir, exist_ok=True)
# torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))
#
# print('Generating new samples...')
# rnd_samples, _ = model.sample(12*12, with_noise=False)
# rnd_samples = rnd_samples.cpu().numpy().reshape(-1, 3, w, w).transpose([0, 2, 3, 1]).reshape(-1, w*w*3)
# mosaic = samples_to_mosaic(rnd_samples, image_shape=[w, w, 3])
# imwrite(os.path.join(model_dir, 'samples.jpg'), mosaic)
