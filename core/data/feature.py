import os.path
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import local_binary_pattern, graycomatrix, multiscale_basic_features, SIFT
from core.util.constants import FEATURE_DIR_PATH, IMGDIR_PATH, DATASET_PATH


class FeatureExtractor:
    def __init__(self,
                 img_dir_path=IMGDIR_PATH,
                 feature_dir_path=FEATURE_DIR_PATH):
        self.save_path = feature_dir_path
        self.img_path = img_dir_path

    def pre_process(self, df: pd.DataFrame, feature: str, should_bb=True, should_resize=False, **kwargs):
        dir_new = self.dirpath_from_ft(feature)
        if not os.path.exists(dir_new):
            os.makedirs(dir_new)
        ft = getattr(self, feature)
        for index, row in df.iterrows():
            f_name = row['path'].split("/")[-1]
            p_new = dir_new + f_name
            if not os.path.exists(p_new):
                with Image.open(row['path']) as img:
                    if feature == "lbp":
                        lbp_arr = ft(img, kwargs.get('method', 'ror'), kwargs.get('radius', 1))
                        img = Image.fromarray(lbp_arr)
                        df.at[index, feature] = p_new
                        print(row)
                    elif feature == "homsc":
                        homsc_arr = np.array(ft(img))
                        # todo should save in json, csv or something else
                        df.at[index, feature] = p_new
                    elif feature == "sift":
                        sift_arr = ft(img)
                        # todo should save in json, csv or something else
                    if should_bb:
                        img = self.make_square_with_bb(img)
                    if should_resize:
                        img = img.resize((416, 416))
                    img.save(p_new)
            else:
                df.at[index, feature] = p_new
        df.drop(df[df.path == ""].index, inplace=True)
        df.to_csv(DATASET_PATH, index=False)
        return df

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
    def homsc(img: Image.Image):
        img = img.convert("L")
        res = np.array(multiscale_basic_features(np.array(img), num_workers=2, sigma_min=1, sigma_max=15))
        print(res.shape)
        return res

    @staticmethod
    def sift(img: Image.Image):
        sift_detector = SIFT()
        img = img.convert("L")
        sift_detector.detect_and_extract(img)
        print(sift_detector.keypoints)
        print(sift_detector.descriptors)
        for keypoint in sift_detector.keypoints:
            pass
        pass

    @staticmethod
    def make_square_with_bb(im, min_size=256, fill_color=(0, 0, 0, 0), mode="RGB"):
        x, y = im.size
        size = max(min_size, x, y)
        new_im = Image.new(mode, (size, size), fill_color)
        new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    @staticmethod
    def make_image_degrees(img: Image.Image, name: str, img_path: str, degrees: list = ["all"]):
        """Creates transformed images from input image, can rotate and flip
        @param img: image to transform
        @param name: name of image file without file type
        @param img_path: path to species folder
        @param degrees: list of strings, include "rotate" to rotate, "flip" to flip, "all" is default for both
        """
        new_images = [img]
        if "all" in degrees or "rotate" in degrees:
            for i in range(3):
                new_images.append(img.rotate((i + 1) * 90, expand=True))
        if "all" in degrees or "flip" in degrees:
            temp_list = []
            for im in new_images:
                temp_list.append(im.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT))
            new_images.extend(temp_list)
        # Only rotations (no flips)
        if len(new_images) == 4:
            for i in range(4):
                # im.show()
                new_images[i].save(f"{img_path}/{name}_{i * 90}.jpg")
                i += 1

        # Rotations and flips, or just flips
        if len(new_images) == 2 or len(new_images) == 8:
            for i in range(int(len(new_images) / 2)):
                # new_images[i].show()
                new_images[i].save(f"{img_path}/{name}_{i * 90}.jpg")
            for i in range(int(len(new_images) / 2), int(len(new_images))):
                # new_images[i].show()
                new_images[i].save(f"{img_path}/{name}_{(i - int(len(new_images) / 2)) * 90}f.jpg")
