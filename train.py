import sys, os
import torch
from torchvision.datasets import CelebA, MNIST
import torchvision.transforms as transforms
import numpy as np
from mfa import MFA
from utils import CropTransform, ReshapeTransform, samples_to_mosaic, visualize_model
from matplotlib import pyplot as plt
from imageio import imwrite
from packaging import version

"""
MFA model training (data fitting) example.
Note that actual EM (and SGD) training code are part of the MFA class itself.
"""


def main(argv):
    assert version.parse(torch.__version__) >= version.parse('1.2.0')

    dataset = argv[1] if len(argv) == 2 else 'celeba'
    print('Preparing dataset and parameters for', dataset, '...')

    if dataset == 'celeba':
        image_shape = [64, 64, 3]       # The input image shape
        n_components = 300              # Number of components in the mixture model
        n_factors = 10                  # Number of factors - the latent dimension (same for all components)
        batch_size = 1000               # The EM batch size
        num_iterations = 30             # Number of EM iterations (=epochs)
        feature_sampling = 0.2          # For faster responsibilities calculation, randomly sample the coordinates (or False)
        mfa_sgd_epochs = 0              # Perform additional training with diagonal (per-pixel) covariance, using SGD
        init_method = 'rnd_samples'   # Initialize each component from few random samples using PPCA
        trans = transforms.Compose([CropTransform((25, 50, 25+128, 50+128)), transforms.Resize(image_shape[0]),
                                    transforms.ToTensor(),  ReshapeTransform([-1])])
        train_set = CelebA(root='./data', split='train', transform=trans, download=True)
        test_set = CelebA(root='./data', split='test', transform=trans, download=True)
    elif dataset == 'mnist':
        image_shape = [28, 28]          # The input image shape
        n_components = 50               # Number of components in the mixture model
        n_factors = 6                   # Number of factors - the latent dimension (same for all components)
        batch_size = 1000               # The EM batch size
        num_iterations = 30             # Number of EM iterations (=epochs)
        feature_sampling = False       # For faster responsibilities calculation, randomly sample the coordinates (or False)
        mfa_sgd_epochs = 0              # Perform additional training with diagonal (per-pixel) covariance, using SGD
        init_method = 'kmeans'         # Initialize by using k-means clustering
        trans = transforms.Compose([transforms.ToTensor(),  ReshapeTransform([-1])])
        train_set = MNIST(root='./data', train=True, transform=trans, download=True)
        test_set = MNIST(root='./data', train=False, transform=trans, download=True)
    else:
        assert False, 'Unknown dataset: ' + dataset

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_dir = './models/'+dataset
    os.makedirs(model_dir, exist_ok=True)
    figures_dir = './figures/'+dataset
    os.makedirs(figures_dir, exist_ok=True)
    model_name = 'c_{}_l_{}_init_{}'.format(n_components, n_factors, init_method)

    print('Defining the MFA model...')
    model = MFA(n_components=n_components, n_features=np.prod(image_shape), n_factors=n_factors,
                init_method=init_method).to(device=device)

    print('EM fitting: {} components / {} factors / batch size {} ...'.format(n_components, n_factors, batch_size))
    ll_log = model.batch_fit(train_set, test_set, batch_size=batch_size, max_iterations=num_iterations,
                             feature_sampling=feature_sampling)

    if mfa_sgd_epochs > 0:
        print('Continuing training using SGD with diagonal (instead of isotropic) noise covariance...')
        model.isotropic_noise = False
        ll_log_sgd = model.sgd_mfa_train(train_set, test_size=256, max_epochs=mfa_sgd_epochs,
                                         feature_sampling=feature_sampling)
        ll_log += ll_log_sgd

    print('Saving the model...')
    torch.save(model.state_dict(), os.path.join(model_dir, 'model_'+model_name+'.pth'))

    print('Visualizing the trained model...')
    model_image = visualize_model(model, image_shape=image_shape, end_component=10)
    imwrite(os.path.join(figures_dir, 'model_'+model_name+'.jpg'), model_image)

    print('Generating random samples...')
    rnd_samples, _ = model.sample(100, with_noise=False)
    mosaic = samples_to_mosaic(rnd_samples, image_shape=image_shape)
    imwrite(os.path.join(figures_dir, 'samples_'+model_name+'.jpg'), mosaic)

    print('Plotting test log-likelihood graph...')
    plt.plot(ll_log, label='c{}_l{}_b{}'.format(n_components, n_factors, batch_size))
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, 'training_graph_'+model_name+'.jpg'))
    print('Done')

if __name__ == "__main__":
    main(sys.argv)
