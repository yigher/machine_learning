"""script to process images into a format compatible with the retrain.py"""
import csv
import os


class LabelsToImageDirectoryProcessor(object):
    """LabelsToImageDirectoryProcessor takes images from input_img_dir
    and saves it to the corresponding label directory"""
    def __init__(
            self,
            input_img_dir=None,
            img_name_to_label_csv=None,
            csv_delim=',',
            img_format="jpg"):
        if input_img_dir is None or img_name_to_label_csv is None:
            raise TypeError("mandatory variables input_img_dir and img_name_to_label_csv required.")
        if not os.path.isdir(input_img_dir):
            raise TypeError(input_img_dir, " is not a directory.")
        if not os.path.isfile(img_name_to_label_csv):
            raise TypeError(img_name_to_label_csv, " is not a file.")
        self.input_img_dir = input_img_dir
        self.img_name_to_label_csv = img_name_to_label_csv
        self.csv_delim = csv_delim
        self.img_format = img_format


    def process(self):
        """process the images"""
        header_row = True
        id_idx = None
        breed_idx = None
        with open(self.img_name_to_label_csv) as csv_file:
            read_csv = csv.reader(csv_file, delimiter=self.csv_delim)
            for row in read_csv:
                if header_row:
                    id_idx = [i for i, x in enumerate(row) if x == "id"]
                    breed_idx = [i for i, x in enumerate(row) if x == "breed"]
                    if len(id_idx) > 0:
                        id_idx = id_idx[0]
                    if len(breed_idx) > 0:
                        breed_idx = breed_idx[0]
                    header_row = False
                    continue
                if breed_idx is None or id_idx is None:
                    raise TypeError("csv headers are not valid ")
                file_name = row[id_idx] + '.' + self.img_format
                src_file_name = os.path.join(self.input_img_dir, file_name)
                if not os.path.isfile(src_file_name):
                    print("File does not exist. Skipping....", src_file_name)
                    continue
                dst_dir = os.path.join(self.input_img_dir, row[breed_idx])
                if not os.path.isdir(dst_dir):
                    print("Creating new directory: ", dst_dir)
                    os.mkdir(dst_dir)
                os.rename(src_file_name, os.path.join(dst_dir, file_name))

if __name__ == "__main__":
    PROC = LabelsToImageDirectoryProcessor("data/train", "data/labels.csv/labels.csv")
    PROC.process()
