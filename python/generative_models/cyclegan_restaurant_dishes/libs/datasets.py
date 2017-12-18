"""Creative Applications of Deep Learning w/ Tensorflow.
Kadenze, Inc.
Copyright Parag K. Mital, June 2016.
"""
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import os

def get_image_np_array(path='./train'):
    """Attempt to load the files of the CELEB dataset.
    Parameters
    ----------
    path : str, optional
        Directory where the aligned/cropped dog dataset can be found.

    Returns
    -------
    files : list
        List of file paths to the dataset.
    """
    if not os.path.exists(path):
        print('Could not find dog dataset under {}.'.format(path))
        return None
    else:
        fs = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    fs.append(os.path.join(root, file))

        fs.sort()
        print("sorted image files. Checking first 10")

        imgs = []
        for f in fs:
            img = plt.imread(f)
            img = img[:, :, :3]
            img = img / 255.0
            imgs.append(img)
        imgs = np.array(imgs)
        return imgs
