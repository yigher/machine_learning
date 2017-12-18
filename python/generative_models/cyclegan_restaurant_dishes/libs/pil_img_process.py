from PIL import Image
from PIL import ImageFilter
import os

def get_filename(full_path):
    """get file name from full path"""
    return full_path.split("/")[-1].split("\\")[-1]

def get_pil_pencilsketch_from_img(file_name):
    """get PIL pencil sketch image"""
    im = Image.open(file_name)
    imout = im.filter(ImageFilter.CONTOUR)
    return imout

def save_pil_img(pil_img, dst):
    """save PIL image"""
    pil_img.save(dst)

def create_pencilsketch_from_imgs(src_dir, dst_dir):
    """save pencil sketch to dst_dir based on images in the src_dir"""
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                img_file = (os.path.join(root, file))
                sketch_img = get_pil_pencilsketch_from_img(img_file)
                save_pil_img(sketch_img, os.path.join(dst_dir, file))

    return True

