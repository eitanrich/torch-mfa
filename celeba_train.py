import os
import torch
from torchvision.datasets import CelebA
import torchvision.transforms as transforms
from mfa import MFA
from utils import CropTransform, ReshapeTransform, samples_to_mosaic, visualize_model
from matplotlib import pyplot as plt
from imageio import imwrite

image_size = 64                 # The image width and height (assumed equal here)
n_components = 300              # Number of components in the mixture model
n_factors = 10                  # Number of factors - the latent dimension (same for all components)
batch_size = 1000               # The EM batch size
num_iterations = 2             # Number of EM iterations (=epochs)
responsibility_sampling = 0.2   # For faster responsibilities calculation, randomly sample the coordinates

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_dir = './models/celeba'
os.makedirs(model_dir, exist_ok=True)

print('Preparing the dataset...')
trans = transforms.Compose([CropTransform((25, 50, 25+128, 50+128)), transforms.Resize(image_size),
                            transforms.ToTensor(),  ReshapeTransform([-1])])
train_set = CelebA(root='data/CelebA', split='train', transform=trans, download=True)
test_set = CelebA(root='data/CelebA', split='test', transform=trans, download=True)

print('Defining the MFA model...')
model = MFA(n_components=n_components, n_features=image_size*image_size*3, n_factors=n_factors).to(device=device)

print('EM fitting: {} components / {} factors / batch size {} ...'.format(n_components, n_factors, batch_size))
ll_log = model.batch_fit(train_set, test_set, batch_size=batch_size, max_iterations=num_iterations,
                         responsibility_sampling=responsibility_sampling)

print('Saving model...')
torch.save(model.state_dict(), os.path.join(model_dir, 'model_c_{}_l_{}.pth'.format(n_components, n_factors)))

print('Visualizing the trained model...')
model_image = visualize_model(model, image_shape=[image_size, image_size, 3], end_component=10)
imwrite(os.path.join(model_dir, 'model_c_{}_l_{}.jpg'.format(n_components, n_factors)), model_image)

print('Generating random samples...')
rnd_samples, _ = model.sample(100, with_noise=False)
mosaic = samples_to_mosaic(rnd_samples, image_shape=[image_size, image_size, 3])
imwrite(os.path.join(model_dir, 'samples_c_{}_l_{}.jpg'.format(n_components, n_factors)), mosaic)

print('Plotting test log-likelihood graph...')
plt.plot(ll_log, label='c{}_l{}_b{}'.format(n_components, n_factors, batch_size))
plt.grid(True)
plt.savefig(os.path.join(model_dir, 'training_graph_c_{}_l_{}.jpg'.format(n_components, n_factors)))
print('Done')
