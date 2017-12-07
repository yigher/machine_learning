"""script to train cyclegan"""
from libs.cyclegan import train
from libs.datasets import get_image_np_array

x_dir = 'imgs/x'
y_dir = 'imgs/y'

x_imgs = get_image_np_array(x_dir)
y_imgs = get_image_np_array(y_dir)

train(x_imgs, y_imgs, shuffle=True)
