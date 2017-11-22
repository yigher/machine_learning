"""main"""
import matplotlib.pyplot as plt
import numpy as np
from libs.utils import get_images, split_image
from libs.mgd_fc import MGD_NeuralNet
from scipy.misc import imresize

IMG_SIZE = (128, 128)
file_n = 100
files = get_images('img_shoes')
files = files[:file_n]
xs = []
ys = []
print('converting image files to array')
for i, f in enumerate(files):
    img = plt.imread(f)
    # img = imresize(img, IMG_SIZE)
    x, y = split_image(img)
    xs.append(x)
    ys.append(y)

xs = np.array(xs)
xs = np.array(xs).reshape(-1, 2)
ys = np.array(ys)
ys = np.array(ys).reshape(-1, 3)

print('normalising xs')
xs = (xs - np.mean(xs)) / np.std(xs)
print('normalising ys')
ys = (ys / 255.0)

print(xs.shape)
print(ys.shape)

net = MGD_NeuralNet()
print('training')
net.train(xs=xs, ys=ys, img_shape=img.shape)
