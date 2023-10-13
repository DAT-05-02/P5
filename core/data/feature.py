import os.path
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import local_binary_pattern, graycomatrix
from core.util.constants import FEATURE_DIR_PATH, IMGDIR_PATH


class FeatureExtractor:
    def __init__(self,
                img_dir_path=IMGDIR_PATH,
                feature_dir_path=FEATURE_DIR_PATH):
        self.save_path = feature_dir_path
        self.img_path = img_dir_path

    def pre_process(self, df: pd.DataFrame, feature: str, should_bb=True, **kwargs):
        dir_new = self.dirpath_from_ft(feature)
        if not os.path.exists(dir_new):
            os.makedirs(dir_new)
        ft = getattr(self, feature)
        for index, row in df.iterrows():
            print(row)
            f_name = row['path'].split("/")[-1]
            p_new = dir_new + f_name
            with Image.open(row['path']) as img:
                if feature == "lbp":
                    lbp_arr = ft(img, kwargs.get('method', 'ror'), kwargs.get('radius', 1))
                    img = Image.fromarray(lbp_arr)
                    if should_bb:
                        self.make_square_with_bb(img)
                    img.save(p_new)
                    df.at[index, feature] = p_new

        return dir_new

    def dirpath_from_ft(self, feature):
        return f"{self.save_path}_{feature}/"

    @staticmethod
    def lbp(img: Image, method="ror", radius=1):
        """Create Local Binary Pattern for an image
        @param img: image to convert
        @param method: which method, accepts 'default', 'ror', 'uniform', 'nri_uniform' or 'var'.
        Read skimage.feature.local_binary_pattern for more information
        @param radius: how many pixels adjacent to center pixel to calculate from.
        @return: (n, m) array as image
        """
        n_points = 8 * radius
        if img.mode != "L":
            img = img.convert("L")

        return np.array(local_binary_pattern(img, n_points, radius, method), dtype=np.uint8)


    @staticmethod
    def rlbp(img: Image, method="ror", radius=1):
        """Create RGB Local Binary Pattern for an image. Instead of greyscale, creates LBP for each RGB channel
        @param img: image to convert
        @param method: which method, accepts 'default', 'ror', 'uniform', 'nri_uniform' or 'var'.
        Read skimage.feature.local_binary_pattern for more information
        @param radius: how many pixels adjacent to center pixel to calculate from.
        @return: (n, m, 3) array as image
        """
        n_points = 8 * radius
        channels = [img.getchannel("R"), img.getchannel("G"), img.getchannel("B")]
        return (local_binary_pattern(ch, n_points, radius, method) for ch in channels)

    @staticmethod
    def glcm(img: Image, distance: list, angles: list):
        if angles is None:
            angles = range(0, 361, 45)
        if distance is None:
            distance = range(0, 5)
        if img.mode != "L":
            img = img.convert("L")
        return graycomatrix(img, distance, angles)

    @staticmethod
    def make_square_with_bb(im, min_size=256, fill_color=(0, 0, 0, 0), mode="RGB"):
        x, y = im.size
        size = max(min_size, x, y)
        new_im = Image.new(mode, (size, size), fill_color)
        new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
        return new_im
