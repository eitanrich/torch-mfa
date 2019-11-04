import torch
import numpy as np
from mfa import MFA


class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


class CropTransform:
    def __init__(self, bbox):
        self.bbox = bbox

    def __call__(self, img):
        return img.crop(self.bbox)


def samples_to_np_images(samples, image_shape=[64, 64, 3], clamp=True):
    assert len(samples.shape) == 2
    assert samples.shape[1] == np.prod(image_shape)
    assert len(image_shape) == 2 or (len(image_shape) == 3 and image_shape[2] > 1)
    samples_out = samples if not clamp else torch.clamp(samples, 0., 1.)
    if len(image_shape) == 3:
        return samples_out.reshape(-1, image_shape[2], image_shape[0], image_shape[1]).permute(0, 2, 3, 1).cpu().numpy()
    else:
        return samples_out.reshape(-1, image_shape[0], image_shape[1]).cpu().numpy()


def sample_to_np_image(sample, image_shape=[64, 64, 3]):
    return samples_to_np_images(sample.unsqueeze(0), image_shape).squeeze()


def samples_to_mosaic(samples, image_shape=[64, 64, 3]):
    images = samples_to_np_images(samples, image_shape)
    num_images = images.shape[0]
    num_cols = int(np.ceil(np.sqrt(num_images)))
    rows = []
    for i in range(num_images // num_cols):
        rows.append(np.hstack([images[j] for j in range(i*num_cols, (i+1)*num_cols)]))
    return np.vstack(rows)


def visualize_model(model: MFA, image_shape=[64, 64, 3], start_component=0, end_component=None):
    assert len(image_shape) == 2 or (len(image_shape) == 3 and image_shape[2] > 1)
    K, d, l = model.A.shape
    h, w = image_shape[:2]
    spacer = min(8, w//8)
    end_component = end_component or min(K, 2048//(w*3+2+spacer))
    k = end_component - start_component
    z = 1.5

    def to_im(x):
        return sample_to_np_image(x, image_shape=image_shape)

    if len(image_shape) == 3:
        canvas = np.ones([(l+1)*(h+1), k*(w*3+2) + (k-1)*spacer, image_shape[2]])
    else:
        canvas = np.ones([(l+1)*(h+1), k*(w*3+2) + (k-1)*spacer])
    for c_num in range(start_component, end_component):
        x_start = (c_num-start_component)*(w*3+2+spacer)

        mu = model.MU[c_num]
        canvas[:h, x_start+w//2:x_start+w//2+w] = to_im(mu)

        D = torch.exp(0.5*model.log_D[c_num])
        canvas[:h, x_start+w//2+w+2:x_start+w//2+2*w+2] = to_im(D / torch.max(D))

        for i in range(l):
            y_start = (i+1)*(h+1)
            A_i = model.A[c_num, :, i]
            canvas[y_start:y_start+h, x_start:x_start+w] = to_im(mu + z * A_i)
            canvas[y_start:y_start+h, x_start+w+1:x_start+2*w+1] = to_im(0.5 + z * A_i)
            canvas[y_start:y_start+h, x_start+2*w+2:x_start+3*w+2] = to_im(mu - z * A_i)
    return canvas
