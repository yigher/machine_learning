import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize as imresize

def preprocess(img, crop=True, resize=True, dsize=(256, 256)):
    if img.dtype != np.uint8:
        img *= 255.0

    if crop:
        crop = np.min(img.shape[:2])
        r = (img.shape[0] - crop) // 2
        c = (img.shape[1] - crop) // 2
        cropped = img[r: r + crop, c: c + crop]
    else:
        cropped = img

    if resize:
        rsz = imresize(cropped, dsize, preserve_range=True)
    else:
        rsz = cropped

    if rsz.ndim == 2:
        rsz = rsz[..., np.newaxis]

    rsz = rsz.astype(np.uint8)
    # subtract imagenet mean
    return (rsz)

path = "imgs/food_orig"
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.jpg'):
            f = os.path.join(root, file)
            img = plt.imread(f)
            img = preprocess(img)
            print(f, "-", "img.shape: ", img.shape)
            plt.imsave(f, img)
