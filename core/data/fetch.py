import concurrent
import logging
from concurrent.futures import ThreadPoolExecutor, ALL_COMPLETED
from typing import Any

import numpy as np
import requests
import pandas as pd
import os
from PIL import Image
from ultralytics import YOLO
from urllib3.exceptions import InsecureRequestWarning
import urllib3

from core.util.logging.logable import Logable
from core.util.util import timing, setup_log, log_ent_exit
from core.util.constants import IMGDIR_PATH, MERGE_COLS, BFLY_FAMILY, BFLY_LIFESTAGE, DATASET_PATH, DIRNAME_DELIM, RAW_WORLD_DATA_PATH, RAW_WORLD_LABEL_PATH
from core.data.feature import FeatureExtractor
from core.yolo.yolo_func import obj_det, yolo_crop

import math

urllib3.disable_warnings(category=InsecureRequestWarning)


class Database(Logable):
    def __init__(self,
                 raw_dataset_path: str,
                 raw_label_path: str,
                 label_dataset_path: str,
                 dataset_csv_filename: str,
                 bfly: list,
                 ft_extractor: FeatureExtractor,
                 degrees: str,
                 link_col="identifier",
                 num_rows=None,
                 crop=True,
                 minimum_images=None,
                 sort=False,
                 log_level=logging.INFO):
        """ Database class
            @param raw_label_path: path to original label dataset file
            @param raw_dataset_path: path to original dataset file
            @param label_dataset_path: path to label csv file
            @param dataset_csv_filename: filename for the csv file
            @param link_col: name of column containing links
            @param num_rows: number of rows to include
            @param sort: if dataset should be sorted by species
            @param bfly: list of species that is included in dataset, have "all" in list for only butterflies (no moths)
            @param crop: boolean if yolo should run and crop images
            @param minimum_images: number of minimum images per species, if no minimum put None
            """
        super().__init__()
        setup_log(log_level=log_level)
        self.raw_dataset_path = raw_dataset_path
        self.raw_label_path = raw_label_path
        self.label_dataset_path = label_dataset_path
        self.dataset_csv_filename = dataset_csv_filename
        self.link_col = link_col
        self.ft_extractor = ft_extractor
        self.bfly = bfly
        self.degrees = degrees
        self._num_rows = num_rows
        self.crop = crop
        self.minimum_images = minimum_images
        self.num_rows = self.num_rows()
        self.sort = sort

    def setup_dataset(self):
        """Constructs dataframe. Makes necessary directories, checks if current .csv file fits with self, and returns
        if so. If not, constructs a new dataframe based on raw data files. Then merges labels and links, drops
        redundant columns and rows and calls to fetch the images in the resulting dataframe. Finally, saves the csv file
        and returns the dataframe.
        @return:
        """
        self._make_img_dir()
        df = self._csv_fits_self()
        if df is not None:
            return df

        if os.path.exists(self.dataset_csv_filename):
            os.remove(self.dataset_csv_filename)

        df: pd.DataFrame = pd.read_csv(self.raw_dataset_path, sep="	", low_memory=False)
        if os.path.exists(self.label_dataset_path):
            df_label: pd.DataFrame = pd.read_csv(self.label_dataset_path, low_memory=False)
        else:
            df_label: pd.DataFrame = pd.read_csv(self.raw_label_path, sep="	", low_memory=False)
            df_label.to_csv(self.label_dataset_path, index=False)
        print(df[df.columns])
        self.drop_cols([df, df_label])
        df = df.merge(df_label[df_label['gbifID'].isin(df['gbifID'])], on=['gbifID'])
        df = df.loc[~df['lifeStage'].isin(BFLY_LIFESTAGE)]
        df = df.dropna(subset=['species'])
        if self.bfly:
            if "all" in self.bfly:
                df = df.loc[df['family'].isin(BFLY_FAMILY)]
            else:
                df = df.loc[df['species'].isin(self.bfly)]
        print(df.shape)
        if self.sort:
            df.sort_values(by=['species'], inplace=True)
        print(f"found {len(df['species'].unique())} unique species")

        # The full danish dataset
        df_dk = df.copy()

        if self._num_rows:
            df.drop(df.index[self._num_rows:], inplace=True)
        df.reset_index(inplace=True, drop=True)

        df_dk.drop(df_dk[df_dk['gbifID'].isin(df['gbifID'])].index, inplace=True)

        df, df_label = self._make_dfs_from_raws()
        df = self._merge_dfs_on_gbif(df, df_label)
        df = self._sort_drop_rows(df)

        if self.minimum_images is not None:
            species_n = df['species'].nunique()
            df = self.pad_dataset(df, df_dk, species_n, RAW_WORLD_DATA_PATH, RAW_WORLD_LABEL_PATH)

        df = self.fetch_images(df, self.link_col)
        df.reset_index(inplace=True, drop=True)
        df.to_csv(self.dataset_csv_filename, index=False)
        return df

    @staticmethod
    def pad_dataset(df, df_dk, species_n, raw_dataset_path_world: str, raw_label_path_world: str):

        values = df['species'].value_counts().keys().tolist()
        counts = df['species'].value_counts().tolist()

        less_than_list = []

        total_rows = len(df.index)

        min_amount_of_pictures = math.floor(total_rows / species_n)

        for itt in range(len(counts)):
            less_than_list.append((values[itt], counts[itt]))

        world_df: pd.DataFrame = pd.read_csv(raw_dataset_path_world, sep="	", low_memory=False)
        world_df_labels: pd.DataFrame = pd.read_csv(raw_label_path_world, sep=",", low_memory=False)

        Database.drop_cols([world_df, world_df_labels])

        world_df = world_df.merge(world_df_labels[world_df_labels['gbifID'].isin(world_df['gbifID'])],
                                  on=['gbifID'])

        world_df = pd.concat((df_dk, world_df))

        out = df["species"].value_counts()

        species_with_less_than_optimal_amount_of_images = []

        rows_filled = 0
        for index, count in out.items():
            species_with_less_than_optimal_amount_of_images.append(index)
            rows_filled += 1

        df = df[0:0]

        rows_filled = rows_filled * min_amount_of_pictures
        rows_to_fill = total_rows - rows_filled

        world_df = world_df.loc[world_df['species'].isin(species_with_less_than_optimal_amount_of_images)]

        # loop that gets the species, which are below the required amount
        for item, count in less_than_list:
            world_specific = world_df.loc[world_df["species"] == item]
            if rows_to_fill > 0:
                world_specific = world_specific.iloc[:min_amount_of_pictures + 1]
                rows_to_fill = rows_to_fill - 1
            else:
                world_specific = world_specific.iloc[:min_amount_of_pictures]

            df = pd.concat((df, world_specific))

        return df

    @timing
    def fetch_images(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Fetches all image links in a DataFrame column to path defined by :func:`~fetch.img_path_from_row`
        and assigns the 'path' column value to saved location. Creates augmented images defined by 'degrees' argument,
        and saves the df to a csv file.
        @param df: the DataFrame containing links
        @param col: which column the links are in
        """
        paths = np.full(len(df.index), fill_value=np.nan).tolist()
        yolo_accepted = np.full(len(df.index), fill_value=np.nan).tolist()

        def save_img(row: pd.Series, index) -> Any:
            """ThreadPoolExecutor function, if file exists, returns the path. Tries to download, extract YOLO prediction,
            and if this returns a found butterfly, inserts black bars, resizes and saves. Returns the saved path.
            @param row: with download link
            @param index: base name of file
            @return: saved path of .npy file
            """
            path = self.img_path_from_row(row, index)
            out = np.nan
            accepted = False
            if not os.path.exists(path):
                try:
                    img = Image.open(requests.get(row[col], stream=True, timeout=40, verify=False).raw)
                    if self.crop is True:
                        model = YOLO('yolo/medium250e.pt')
                        res = obj_det(img, model, conf=0.50)
                        xywhn = res[0].boxes.xywhn
                        if xywhn.numel() > 0:
                            img = yolo_crop(img, xywhn)
                            accepted = True
                    img = FeatureExtractor.make_square_with_bb(img)
                    img = img.resize((416, 416))
                    img = np.asarray(img)
                    np.save(path, img, allow_pickle=True)
                    out = path
                    self.info(out)
                except requests.exceptions.Timeout:
                    self.debug(f"Timeout occurred for index {index}")
                except requests.exceptions.RequestException as e:
                    self.debug(f"Error occurred: {e}")
                except ValueError as e:
                    self.error(f"Image name not applicable: {e}")
                    raise e
                except OSError as e:
                    self.error(f"Could not save file: {e}")
                    raise e
                except Exception as e:
                    self.error(f"Unknown error: {e}")
                    raise e
            else:
                out = path
                accepted = True
            return out, accepted

        with ThreadPoolExecutor(50) as executor:
            """Iterates through all rows, starts a thread to download, yolo-predict, save etc. each individual row, 
            then collects all results in a list, and insert these as a column 'path'. Drops rows with NaN value to get 
            rid of failed downloads, no yolo result or any uncaught exception"""
            futures = [executor.submit(save_img, row, index) for index, row in df.iterrows()]
            concurrent.futures.wait(futures, timeout=None, return_when=ALL_COMPLETED)
            for i, ft in enumerate(futures):
                paths[i], yolo_accepted[i] = ft.result()

            df['path'] = paths
            df['yolo_accepted'] = yolo_accepted
            df.dropna(subset=['path', 'yolo_accepted'], inplace=True)
        df = self.ft_extractor.create_augmented_df(df=df, degrees=self.degrees)
        self.info(df)
        df.to_csv(DATASET_PATH, index=False)
        return df

    def _csv_fits_self(self):
        """Checks if .csv file exists returns if aggregated number of rows fits with the read file.
        @return: df if compliant, otherwise None
        """
        if os.path.exists(self.dataset_csv_filename):
            df = pd.read_csv(self.dataset_csv_filename, low_memory=False)
            self.debug(f"len(df): {len(df)}, self.num_rows: {self.num_rows}")
            if self._num_rows and len(df) == self.num_rows:
                return df
        return None

    def _merge_dfs_on_gbif(self, df: pd.DataFrame, df_label: pd.DataFrame):
        """Merges 2 dataframes into one given the gbifID. Essentially connects links with species and other relevant
        information. Removes moths from resulting dataframe, or based on self.bfly containing butterfly families
        to be included.
        @param df: link dataframe
        @param df_label: label dataframe
        @return: merged dataframe
        """
        df = df.merge(df_label[df_label['gbifID'].isin(df['gbifID'])], on=['gbifID'])
        df = df.loc[~df['lifeStage'].isin(BFLY_LIFESTAGE)]
        df = df.dropna(subset=['species'])
        if self.bfly:
            if "all" in self.bfly:
                df = df.loc[df['family'].isin(BFLY_FAMILY)]
            else:
                df = df.loc[df['species'].isin(self.bfly)]
        self.info(f'df shape: {df.shape}')
        return df

    def _sort_drop_rows(self, df: pd.DataFrame):
        """If self is initialized with sort, sort species. If self has num_rows, drop rest of rows.
        @param df: dataframe
        @return resulting dataframe after sort/drop"""
        if self.sort:
            df.sort_values(by=['species'], inplace=True)
        self.info(f"found {len(df['species'].unique())} unique species")
        if self._num_rows:
            df.drop(df.index[self._num_rows:], inplace=True)
        df.reset_index(inplace=True, drop=True)
        return df

    def _make_dfs_from_raws(self):
        """Reads raw data files for links and labels into dataframes
        @return: tuple of df with links and df with labels
        """
        df: pd.DataFrame = pd.read_csv(self.raw_dataset_path, sep="	", low_memory=False)
        if os.path.exists(self.label_dataset_path):
            df_label: pd.DataFrame = pd.read_csv(self.label_dataset_path, low_memory=False)
        else:
            df_label: pd.DataFrame = pd.read_csv(self.raw_label_path, sep="	", low_memory=False)
            df_label.to_csv(self.label_dataset_path, index=False)
        self.drop_cols([df, df_label])
        self.info(df[df.columns])
        return df, df_label

    @staticmethod
    def img_path_from_row(row: pd.Series, index: int, extra=None):
        """Generates path for an image based on row and index with optional column, in which the image link is.
        @param row: Series to extract file path from
        @param index: of the row. Series objects don't inherently know which index they are in a DataFrame.
        @param extra: (optional) extra keywords in name
        @return: the path to save the image in
        @rtype: str
        """
        try:
            path = f"{IMGDIR_PATH}{row['species'].replace(' ', DIRNAME_DELIM)}"
        except AttributeError as e:
            print(f"{row} error: \n {e}")
            raise e
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        out = f"{IMGDIR_PATH}{row[9].replace(' ', DIRNAME_DELIM)}/{index}"
        if extra:
            out += extra
        return out + "_0.npy"

    @log_ent_exit
    def only_accepted(self, df) -> pd.DataFrame:
        df = df.loc[df['yolo_accepted'].isin(['True', True])]
        return df

    @staticmethod
    def drop_cols(dfs):
        """drops all columns not MERGE_COLS from the given dataframes
        @param dfs: list of dataframes to remove columns for
        """
        for df in list(dfs):
            df.drop(columns=[col for col in df if col not in MERGE_COLS], inplace=True)

    @staticmethod
    def _make_img_dir():
        """Creates image-db folder if not exists"""
        if not os.path.exists(IMGDIR_PATH):
            os.makedirs(IMGDIR_PATH)

    @log_ent_exit
    def num_rows(self):
        """Aggregate number of rows depending on self.degrees.
        @return: aggregated num_rows
        """
        if self._num_rows is None:
            return None
        elif self.degrees == "all":
            return self._num_rows * 8
        elif self.degrees == "flip":
            return self._num_rows * 2
        elif self.degrees == "rotate":
            return self._num_rows * 4
        return self._num_rows
