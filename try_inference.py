import os
import torch
import numpy as np
from torchvision.datasets import CelebA
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler, RandomSampler
from mfa import MFA
from utils import *
from matplotlib import pyplot as plt
from imageio import imwrite
from tqdm import tqdm

if __name__ == "__main__":
    dataset = 'celeba'
    find_outliers = True
    reconstruction = True
    inpainting = True

    print('Preparing dataset and parameters for', dataset, '...')
    if dataset == 'celeba':
        image_shape = [64, 64, 3]       # The input image shape
        n_components = 200              # Number of components in the mixture model
        n_factors = 10                  # Number of factors - the latent dimension (same for all components)
        batch_size = 128                # The EM batch size
        num_iterations = 10             # Number of EM iterations (=epochs)
        responsibility_sampling = 0.2   # For faster responsibilities calculation, randomly sample the coordinates (or False)
        mfa_sgd_epochs = 0              # Perform additional training with diagonal (per-pixel) covariance, using SGD
        trans = transforms.Compose([CropTransform((25, 50, 25+128, 50+128)), transforms.Resize(image_shape[0]),
                                    transforms.ToTensor(),  ReshapeTransform([-1])])
        test_dataset = CelebA(root='./data', split='test', transform=trans, download=True)
        # The train set has more interesting outliers...
        # test_dataset = CelebA(root='./data', split='train', transform=trans, download=True)
    else:
        assert False, 'Unknown dataset: ' + dataset

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_dir = './models/' + dataset
    figures_dir = './figures/' + dataset
    os.makedirs(figures_dir, exist_ok=True)

    print('Loading pre-trained MFA model...')
    model = MFA(n_components=n_components, n_features=np.prod(image_shape), n_factors=n_factors).to(device=device)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model_c_{}_l_{}.pth'.format(n_components, n_factors))))

    if find_outliers:
        print('Finding dataset outliers...')
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        all_ll = []
        for batch_x, _ in tqdm(loader):
            all_ll.append(model.log_prob(batch_x.to(device)))
        all_ll = torch.cat(all_ll, dim=0)
        ll_sorted = torch.argsort(all_ll).cpu().numpy()

        all_keys = [key for key in SequentialSampler(test_dataset)]
        outlier_samples, _ = zip(*[test_dataset[all_keys[ll_sorted[i]]] for i in range(100)])
        mosaic = samples_to_mosaic(torch.stack(outlier_samples), image_shape=image_shape)
        imwrite(os.path.join(figures_dir, 'outliers.jpg'), mosaic)

    if reconstruction:
        print('Reconstructing images from the trained model...')
        random_samples, _ = zip(*[test_dataset[k] for k in RandomSampler(test_dataset, replacement=True, num_samples=100)])
        random_samples = torch.stack(random_samples)

        if inpainting:
            # Hide part of each image
            w = image_shape[0]
            mask = np.ones([3, w, w], dtype=np.float32)
            # mask[:, w//4:-w//4, w//4:-w//4] = 0
            mask[:, :, w//2:] = 0
            mask = torch.from_numpy(mask.flatten()).reshape([1, -1])
            original_full_samples = random_samples.clone()
            random_samples *= mask
            used_features = torch.nonzero(mask.flatten()).flatten()
            reconstructed_samples = model.conditional_reconstruct(random_samples.to(device), observed_features=used_features).cpu()
        else:
            reconstructed_samples = model.reconstruct(random_samples.to(device)).cpu()

        if inpainting:
            reconstructed_samples = random_samples * mask + reconstructed_samples * (1 - mask)

        mosaic_original = samples_to_mosaic(random_samples, image_shape=image_shape)
        imwrite(os.path.join(figures_dir, 'original_samples.jpg'), mosaic_original)
        mosaic_recontructed = samples_to_mosaic(reconstructed_samples, image_shape=image_shape)
        imwrite(os.path.join(figures_dir, 'reconstructed_samples.jpg'), mosaic_recontructed)
