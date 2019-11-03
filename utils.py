import torch
import numpy as np
from mfa import MFA

class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


def sample_to_image(sample, image_shape):
    return sample.reshape(image_shape[2], image_shape[0], image_shape[1]).permute(1, 2, 0)


def samples_to_mosaic(samples, image_shape):
    # TODO: Insert transpose from (ch, h, w) to (h, w, ch) here
    assert samples.shape[1] == np.prod(image_shape)
    num_samples = samples.shape[0]
    num_cols = int(np.ceil(np.sqrt(num_samples)))
    num_rows = num_samples // num_cols
    rows = []
    for i in range(num_rows):
        row_images = [np.minimum(1.0, np.maximum(0.0, samples[j].reshape(image_shape)))
                      for j in range(i*num_cols, (i+1)*num_cols)]
        rows.append(np.hstack(row_images))
    return np.vstack(rows)


def visualize_model(model: MFA, image_shape=[64, 64, 3], start_component=0, end_component=None):
    K, d, l = model.A.shape
    h, w = image_shape[0:2]
    spacer = min(8, w//8)
    end_component = end_component or min(K, 2048//(w*3+2+spacer))
    k = end_component - start_component
    z = 1.5

    def to_im(im):
        return sample_to_image(torch.clamp(im, 0., 1.), image_shape)

    canvas = torch.ones((l+1)*(h+1), k*(w*3+2) + (k-1)*spacer, image_shape[2])
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
    return canvas.cpu().numpy()
