"""script to retrieve z latent data and image data based on directories.
Directory structure has to be classification_name/data_id.[file_extension].
Assumptions: 
z file format - [id].jpg_[model_version].[z_file_ext]
image file format - [id].[img_file_ext]
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from libs.utils import get_dir_delimiter_by_os
from skimage.transform import resize as imresize

class DogDataDao(object):
    """DogDataDao retrieve z latent data and image data based on directories"""
    def __init__(
            self,
            z_dir=None,
            img_dir=None,
            z_file_ext='txt',
            img_file_ext='jpg'):
        self.z_dir = z_dir
        self.img_dir = img_dir
        self.z_file_ext = z_file_ext
        self.img_file_ext = img_file_ext

    def get_dir_labels(self, dir_input):
        """get labels based on dir"""
        if dir_input is None:
            return None
        return [o for o in os.listdir(dir_input)
                if os.path.isdir(os.path.join(dir_input, o))]

    def get_z_labels(self):
        """get z labels based on dir"""
        if self.z_dir is None:
            return None
        return self.get_dir_labels(self.z_dir)

    def get_img_labels(self):
        """get image labels based on dir"""
        if self.img_dir is None:
            return None
        return self.get_dir_labels(self.img_dir)

    def get_z_by_label(self, label_input, num_limit=10, randn_flag=False):
        """get z by label. Return numpy array"""
        file_list = [
            os.path.join(self.z_dir, label_input, o)
            for o in os.listdir(os.path.join(self.z_dir, label_input))
            if o.endswith(self.z_file_ext)]

        if randn_flag:
            file_list = np.random.choice(file_list, num_limit, replace=False)
        else:
            file_list = file_list[:num_limit]

        z_out = np.array([np.fromfile(f, sep=',', dtype=np.float32) for f in file_list])
        return file_list, z_out

    def get_img_by_label(
            self,
            label_input,
            num_limit=10,
            randn_flag=False,
            crop=True, resize=True, dsize=(299, 299)):
        """get images by label. Return numpy array"""
        file_list = [
            os.path.join(self.z_dir, label_input, o)
            for o in os.listdir(os.path.join(self.z_dir, label_input))
            if o.endswith(self.z_file_ext)]

        if randn_flag:
            file_list = np.random.choice(file_list, num_limit, replace=False)
        else:
            file_list = file_list[:num_limit]
        img_out = []
        for f_out in file_list:
            img = plt.imread(f_out)
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
            img_out.append(rsz)
        return file_list, np.array(img_out)

    def get_id_from_file_list(self, file_list):
        """get id from file list"""
        return [get_dir_delimiter_by_os().
                join(f.split(get_dir_delimiter_by_os())[-2:]).split('.')[0] for f in file_list]

    def get_id_from_filename(self, file_name):
        """get id from file list"""
        return get_dir_delimiter_by_os().\
                join(file_name.split(get_dir_delimiter_by_os())[-2:]).split('.')[0]

    def get_img_by_z_filename(
            self,
            z_id,
            crop=True, resize=True, dsize=(299, 299)):
        """get images by label. Return numpy array"""
        z_id = self.get_id_from_filename(z_id)
        file_name = os.path.join(self.img_dir, str(z_id+"."+self.img_file_ext))

        img = plt.imread(file_name)
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
        return file_name, rsz

    def get_img_by_z_filenames(
            self,
            z_ids,
            crop=True, resize=True, dsize=(299, 299)):
        """get images by label. Return numpy array"""
        file_names = []
        imgs = []
        for z_id in z_ids:
            file_name, img = self.get_img_by_z_filename(z_id, crop, resize, dsize)
            file_names.append(file_name)
            imgs.append(img)
        return file_names, np.array(imgs)

    def get_z_by_img_filename(self, img_id):
        """get images by label. Return numpy array"""
        img_id = self.get_id_from_filename(img_id)
        z_subdir, z_file_id = img_id.split(get_dir_delimiter_by_os())
        file_name = None
        for f_out in os.listdir(os.path.join(self.z_dir, z_subdir)):
            if f_out.startswith(z_file_id):
                file_name = os.path.join(self.z_dir, z_subdir, f_out)
                break

        if file_name is None:
            return None, None
        return file_name, np.fromfile(file_name, sep=',', dtype=np.float32)

    def get_z_by_img_filenames(
            self,
            img_ids):
        """get images by label. Return numpy array"""
        file_names = []
        zs_out = []
        for img_id in img_ids:
            file_name, z_out = self.get_z_by_img_filename(img_id)
            file_names.append(file_name)
            zs_out.append(z_out)
        return file_names, np.array(zs_out)

if __name__ == "__main__":
    PROC = DogDataDao("model_out/bottlenecks", "data/train")
    LABELS = PROC.get_z_labels()
    print(LABELS)
    IDX, NP_DATA = PROC.get_z_by_label(LABELS[0], num_limit=2, randn_flag=False)
    print("IDX: ", IDX)
    print("z_out[0]", NP_DATA.shape)
    IMG_FILE_N, IMGS = PROC.get_img_by_z_filenames(IDX)
    print("FILE_N: ", IMG_FILE_N)
    print("IMGS: ", IMGS.shape)
    Z_FILE_N, ZS = PROC.get_z_by_img_filenames(IMG_FILE_N)
    print("Z_FILE_N: ", Z_FILE_N)
    print("ZS: ", ZS.shape)
    