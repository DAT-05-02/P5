import logging
import os.path
import re

import numpy as np
import pandas as pd
import skimage.color
from PIL import Image
from skimage.feature import local_binary_pattern, graycomatrix, SIFT
from core.util.constants import FEATURE_DIR_PATH, IMGDIR_PATH, DATASET_PATH, PATH_SEP, DIRNAME_DELIM
from core.util.logging.logable import Logable
from core.util.util import setup_log, log_ent_exit

FTS = ['sift', 'lbp', 'glcm']


class FeatureExtractor(Logable):
    def __init__(self,
                 img_dir_path=IMGDIR_PATH,
                 feature_dir_path=FEATURE_DIR_PATH,
                 log_level=logging.INFO):
        super().__init__()
        setup_log(log_level=log_level)
        self.save_path = feature_dir_path
        self.img_path = img_dir_path
        self._mk_ft_dirs()

    def pre_process(self,
                    df: pd.DataFrame,
                    feature="",
                    **kwargs):
        """Perform feature extraction on a dataframe's files, saves result in a separate .npy file and insert resulting
        paths into the dataframe. Saves the augmented df.
        @param df: containing paths to .npy arrays to perform feature extraction on.
        @param feature: to extract
        @param kwargs: additional arguments for the given feature
        @return: dataframe with paths to extracted features in separate column
        """
        ft = getattr(self, feature, None)
        paths = np.full(len(df.index), fill_value=np.nan).tolist()
        self.info(f"paths len: {len(paths)}")
        for index, row in df.iterrows():
            p_new = self.path_from_row_ft(row, feature)
            if not os.path.exists(p_new):
                os.makedirs(self.l_dirpath_from_row(row, feature), exist_ok=True)
                img = np.load(row['path'], allow_pickle=True)
                if feature == "lbp":
                    out: np.ndarray = ft(img, kwargs.get('method', 'ror'), kwargs.get('radius', 1))
                    paths[int(index)] = p_new
                elif feature == "sift":
                    out: np.ndarray = ft(img)
                    paths[int(index)] = p_new
                elif feature == "glcm":
                    out: np.ndarray = ft(img, kwargs.get('distances', None), kwargs.get('angles', None))
                    paths[int(index)] = p_new
                elif feature == "" or feature is None:
                    out: np.ndarray = np.array(img)
                else:
                    raise ValueError(f'"{feature}" is not supported')
                np.save(p_new, out, allow_pickle=True)

            else:
                try:
                    paths[int(index)] = p_new
                except IndexError as e:
                    self.error(f"index: {index}")
                    raise e
        df[feature] = paths
        df.to_csv(DATASET_PATH, index=False)
        return df

    def l_dirpath_from_row(self, row: pd.Series, feature: str):
        """Directory path given a label (species) and feature.
        @param row: pd.series to draw data from
        @param feature: name of feature folder
        @return: parent feature path + label path
        """
        try:
            l_name = str(row['path']).split(PATH_SEP)[-2]
            return self.dirpath_from_ft(feature) + l_name + PATH_SEP
        except IndexError as e:
            self.error(f"row: {row}", stack_info=True, exc_info=True)
            raise e

    @log_ent_exit
    def path_from_row_ft(self, row: pd.Series, feature: str):
        """Full path for an individual row
        @param row: pd.Series to draw data from
        @param feature: name of feature folder
        @return: full file path
        """
        try:
            f_name = str(row['path']).split(PATH_SEP)[-1].split('.')[0] + ".npy"
            return self.l_dirpath_from_row(row, feature) + f_name
        except IndexError as e:
            self.error(f"row: {row}", stack_info=True, exc_info=True)
            raise e

    def dirpath_from_ft(self, feature):
        """Outermost parent directory based on feature.
        @param feature: which feature to create dir from
        @return: path of dir
        """
        if feature == "" or feature is None:
            out_ft = "db"
        else:
            out_ft = feature
        return f"{self.save_path}{DIRNAME_DELIM}{out_ft}/"

    @staticmethod
    def lbp(img: np.ndarray, method="ror", radius=1):
        """Create Local Binary Pattern for an image
        @param img: to convert
        @param method: which method, accepts 'default', 'ror', 'uniform', 'nri_uniform' or 'var'.
        Read skimage.feature.local_binary_pattern for more information
        @param radius: how many pixels adjacent to center pixel to calculate from.
        @return: (n, m) array as image
        """
        n_points = 8 * radius
        if len(img.shape) > 2:
            img = skimage.color.rgb2gray(img)

        return local_binary_pattern(img, n_points, radius, method)

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
    def glcm(img: np.ndarray, distance: list, angles: list):
        """Creates Grey-Level-Co-Occurrence Matrix.
        @param img: array of image to convert
        @param distance: depth of matrix, how far away from pixel should it look for co-occurring values
        @param angles: which directions to look
        @return: glcm matrix
        """
        if angles is None:
            angles = range(0, 361, 45)
        if distance is None:
            distance = range(0, 5)
        img = Image.fromarray(img)
        img = img.convert("L")
        img = np.array(img)
        return graycomatrix(img, distance, angles)

    def sift(self, img: Image.Image):
        """Create SIFT features
        @param img: Image to extract features from
        @return: list of tuples of key points and features
        """
        sift_detector = SIFT()
        img = img.convert("L")
        sift_detector.detect_and_extract(img)
        return np.array(list(zip(sift_detector.keypoints, sift_detector.descriptors)))

    @staticmethod
    def make_square_with_bb(im, min_size=56, fill_color=(0, 0, 0, 0), mode="RGB"):
        """Insert black bars around an image, if one dimension is shorter than the other
        @param im: image
        @param min_size: minimum size of image
        @param fill_color: what color (default black) to paste
        @param mode: which mode image resulting image should be
        @return: resulting image with black bars
        """
        x, y = im.size
        size = max(min_size, x, y)
        new_im = Image.new(mode, (size, size), fill_color)
        new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    @log_ent_exit
    def create_augmented_df(self, df: pd.DataFrame, degrees: str = "all") -> pd.DataFrame:
        """Creates transformed image as rows in df inserts them
        @param df: pandas dataframe
        @param degrees: list of strings, include "rotate" to rotate, "flip" to flip, "all" is default for both
        @return: augmented df
        """
        new_rows = []
        for index, row in df.iterrows():
            new_paths = self.augment_image(row, degrees)
            for n in new_paths:
                new_r = row.copy(deep=True)
                new_r['path'] = n
                new_rows.append(new_r)

        df = pd.concat([df, pd.DataFrame(new_rows, columns=df.columns)], ignore_index=True, axis=0)
        return df

    def augment_image(self, row: pd.Series, degrees: str = "all"):
        """Creates transformed images from input image, can rotate and flip
        @param row: pandas series
        @param degrees: list of strings, include "rotate" to rotate, "flip" to flip, "all" is default for both
        @return: list of paths to augmented images
        """
        new_paths = []
        if "all" == degrees:
            new_paths.append(self.flip_and_save_image(row['path']))
            for i in range(90, 360, 90):
                rotated_path = self.rotate_and_save_image(row['path'], i)
                flipped_path = self.flip_and_save_image(rotated_path)
                new_paths.append(rotated_path)
                new_paths.append(flipped_path)
        elif "rotate" == degrees:
            for i in range(1, 4):
                rotated_path = self.rotate_and_save_image(row['path'], i * 90)
                new_paths.append(rotated_path)
        elif "flip" == degrees:
            flipped_path = self.flip_and_save_image(row['path'])
            new_paths.append(flipped_path)
        return new_paths

    @log_ent_exit
    def rotate_and_save_image(self, img_path: str, degree: int) -> str:
        """ Rotates an image 4 and saves the rotated images to a path
        @param img_path: where to store the newly created image
        @param degree: amount of degrees to rotate image
        @return: path to rotated image
        """
        new_path = self.rotate_path(img_path, degree)
        if os.path.exists(new_path):
            return new_path
        try:
            with open(img_path, 'rb') as f:
                image = np.load(f, allow_pickle=True)
                image = Image.fromarray(image)
                image = image.rotate(degree, expand=True)
                np.save(new_path, np.asarray(image))
                return new_path
        except IOError as e:
            print("Error when trying to rotate and save images")
            raise e

    @log_ent_exit
    def rotate_path(self, img_path, degree):
        """Path a rotated image should have
        @param img_path: base path
        @param degree: of image
        @return: path depending on image path and degree
        """
        parts = re.split('[/_.]', img_path)
        self.debug(parts)
        return f"{'/'.join(parts[:3])}_{degree}.{parts[4]}"

    @log_ent_exit
    def flip_and_save_image(self, img_path: str) -> str:
        """ Flips an image and saves to a path
        @param img_path: of image to flip
        @return: path to flipped image
        """
        self.debug(img_path)
        new_path = f"{img_path.split('.')[0]}f.{img_path.split('.')[1]}"
        if os.path.exists(new_path):
            return new_path
        with open(img_path, 'rb') as f:
            try:
                self.debug(img_path)
                image = np.load(f, allow_pickle=True)
                image = Image.fromarray(image)
                image.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
                np.save(new_path, np.asarray(image))
                return new_path
            except IOError as e:
                self.log.error(f"Error when trying to flip and save images: {e}")
                raise e

    @log_ent_exit
    def shape_from_feature(self, df, feature='path', ):
        """Aggregate shape, depending on which feature should be trained on.
        @param df: dataframe
        @param feature: which feature to check shape of
        @raise ValueError: if not all extracted features are of same shape
        @return: shape of feature
        """
        paths = df[feature]
        unique_output_shapes = set()

        for path in paths:
            with open(path, 'rb') as f:
                img = np.load(f)
                output_shape = img.shape
                if output_shape not in unique_output_shapes:
                    self.info(f"unique shape: {output_shape}")
                    self.info(f"at path: {path}")
                    unique_output_shapes.add(output_shape)

        if len(unique_output_shapes) > 1:
            raise ValueError(f"Not all features are the same shape: {unique_output_shapes}")
        return unique_output_shapes.pop()
