import torch
import numpy as np

class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


def samples_to_mosaic(samples, image_shape):
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
