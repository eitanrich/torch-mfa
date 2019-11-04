import os
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from mfa import MFA
from utils import ReshapeTransform, samples_to_mosaic, visualize_model
from imageio import imwrite

model_dir = './models/mnist'

trans = transforms.Compose([transforms.ToTensor(), ReshapeTransform([-1])])
train_set = MNIST(root='./data', train=True, transform=trans, download=True)
test_set = MNIST(root='./data', train=False, transform=trans, download=True)

for n_components in [50, 100]:
    for n_factors in [6, 10]:
        for try_num in range(1):
            model = MFA(n_components=n_components, n_features=28*28, n_factors=n_factors)
            model.cuda()

            print('EM fitting: {} components / {} factors / try {}...'.format(n_components, n_factors, try_num))

            ll_log = model.batch_fit(train_set, test_set, max_iterations=10)
            print('Saving model...')
            torch.save(model.state_dict(), os.path.join(model_dir, 'model_c_{}_l_{}.pth'.format(n_components, n_factors)))
            plt.plot(ll_log)
            plt.grid(True)
            plt.pause(0.1)
            rnd_samples, _ = model.sample(400, with_noise=False)
            mosaic = samples_to_mosaic(rnd_samples, image_shape=[28, 28])
            imwrite(os.path.join(model_dir,
                                 'samples_c_{}_l_{}_try_{}.jpg'.format(n_components, n_factors, try_num)), mosaic)
            model_image = visualize_model(model, image_shape=[28, 28], end_component=10)
            imwrite(os.path.join(model_dir,
                                 'model_c_{}_l_{}_try_{}.jpg'.format(n_components, n_factors, try_num)), model_image)

print('Done')
plt.show()

# plt.savefig('mnist_mppca_online_em_init_with_lx2.pdf')
# print('Saving model...')
# os.makedirs(model_dir, exist_ok=True)
# torch.save(model.state_dict(), os.path.join(model_dir, 'model_batch_1_epoch.pth'))
# torch.save(model.state_dict(), os.path.join(model_dir, 'model_incremental_1_epoch.pth'))

# print('Generating new samples...')
# rnd_samples, _ = model.sample(400, with_noise=False)
# mosaic = samples_to_mosaic(rnd_samples.cpu().numpy(), image_shape=[28, 28])
# imwrite(os.path.join(model_dir, 'samples_batch_1_epoch.jpg'), mosaic)
# imwrite(os.path.join(model_dir, 'samples_incremental_1_epoch.jpg'), mosaic)
