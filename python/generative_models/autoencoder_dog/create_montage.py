from libs.utils import montage
import matplotlib.pyplot as plt
import numpy as np
import os
import imageio

img_dir = 'data/model_imgs/vae20171018_093210'

test_xs = plt.imread(os.path.join(img_dir, 'test_xs.png') )

recon_files = [
    os.path.join(img_dir, f_out) for f_out in os.listdir(img_dir)
    if f_out.startswith('recon') and f_out.endswith('.png')]

all_imgs = []

for f in recon_files:
    print("processing: ", f)
    recon = plt.imread(f)
    all_img = np.hstack((test_xs, recon))
    all_imgs.append(all_img)
print("creating montage")
imageio.mimsave(os.path.join(img_dir, 'montage.gif'), all_imgs)