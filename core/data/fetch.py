import concurrent
import logging
from concurrent.futures import ThreadPoolExecutor, ALL_COMPLETED
from typing import Any

import numpy as np
import requests
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
import urllib3

from core.util.logging.logable import Logable
from core.util.util import timing, setup_log, log_ent_exit, ConstantSingleton
from core.util.constants import (IMGDIR_PATH, MERGE_COLS, BFLY_FAMILY, BFLY_LIFESTAGE, DIRNAME_DELIM,
                                 RAW_WORLD_DATA_PATH, RAW_WORLD_LABEL_PATH)
from core.data.feature import FeatureExtractor
from core.yolo.yolo_func import obj_det, yolo_crop

import math
urllib3.disable_warnings(category=urllib3.exceptions.InsecureRequestWarning)
constants = ConstantSingleton()

ERR_VALUES = ['ERROR', 'TIMEOUT', 'REQUESTEXCEPTION']


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
                 crop=True,
                 num_species=None,
                 num_images=None,
                 log_level=logging.INFO):
        """ Database class
            @param raw_label_path: path to original label dataset file
            @param raw_dataset_path: path to original dataset file
            @param label_dataset_path: path to label csv file
            @param dataset_csv_filename: filename for the csv file
            @param link_col: name of column containing links
            @param num_images: number of images to include
            @param num_species: number of species to include
            @param bfly: list of species that is included in dataset, have "all" in list for only butterflies (no moths)
            @param crop: boolean if yolo should run and crop images
            """
        super().__init__()
        setup_log(log_level=log_level)
        self.raw_dataset_path = raw_dataset_path
        self.raw_label_path = raw_label_path
        self.label_dataset_path = label_dataset_path
        self.dataset_csv_filename = dataset_csv_filename
        self.link_col = link_col
        self.ft_extractor = ft_extractor
        self.num_species = num_species
        self.num_images = num_images
        self.bfly = bfly
        self.degrees = degrees
        self._num_rows = num_images
        self.crop = crop
        self.num_rows = self.num_rows()

    def setup_dataset(self):
        """Constructs dataframe. Makes necessary directories, checks if current .csv file fits with self, and returns
        if so. If not, constructs a new dataframe based on raw data files. Then merges labels and links, drops
        redundant columns and rows and calls to fetch the images in the resulting dataframe. Finally, saves the csv file
        and returns the dataframe.
        @return:
        """

        self._make_img_dir()
        if self._csv_fits_self():
            """Case if current num images/species fits with results. Basically just returns the working window"""
            db = pd.read_csv(self.dataset_csv_filename, low_memory=False)
            df = self._get_working_df(db)
            df.dropna(subset=['path'], inplace=True)
            self.info(db.shape)
            self.info(df.shape)
            return db, df
        elif self._partial_df():
            """Case if we have parts of the required information, merge with more rows from world dataset, and keep 
            the first result if we accidentally merged previously merged rows. The first duplicate will always be the
            one with most information (yolo crop), so keep this and continue as normal"""
            db = pd.read_csv(self.dataset_csv_filename, low_memory=False)
            db = self.pad_dataset(db, RAW_WORLD_DATA_PATH, RAW_WORLD_LABEL_PATH, self.num_images)
            db.drop_duplicates(subset=[self.link_col], keep="first")
            # self.info(df)
        else:
            """Case if we are starting from fresh"""
            db, df_label = self._make_dfs_from_raws()
            db = self._merge_dfs_on_gbif(db, df_label)
            db = self.pad_dataset(db, RAW_WORLD_DATA_PATH, RAW_WORLD_LABEL_PATH, self.num_images)
            db = db.reset_index(drop=True)

        df = self.fetch_images(self._get_working_df(db), self.link_col)
        db = db.merge(df, how="left")
        # update db rows with newly acquired information
        db.update(df)
        db.to_csv(self.dataset_csv_filename, index=False)
        df = df.dropna(subset=['path'])
        df = df.loc[~df['path'].isin(ERR_VALUES)]
        self.info(db.shape)
        self.info(df.shape)
        return db, df

    def pad_dataset(self, df: pd.DataFrame, raw_dataset_path: str, raw_label_path: str, min_amount_of_pictures=3):
        run_correction = False

        values = df['species'].value_counts().keys().tolist()
        counts = df['species'].value_counts().tolist()
        self.info(len(values))
        self.info(counts)

        less_than_list = []

        for itt in range(len(counts)):
            if min_amount_of_pictures > counts[itt]:
                less_than_list.append((values[itt], counts[itt]))
                run_correction = True

        if run_correction:
            world_df: pd.DataFrame = pd.read_csv(raw_dataset_path, sep="	", low_memory=False)
            world_df_labels: pd.DataFrame = pd.read_csv(raw_label_path, sep=",", low_memory=False)

            self.drop_cols([world_df, world_df_labels])

            world_df = world_df.merge(world_df_labels[world_df_labels['gbifID'].isin(world_df['gbifID'])],
                                      on=['gbifID'])
            world_df.dropna(subset=[self.link_col, 'species'])
            out = df["species"].value_counts()

            species_with_less_than_optimal_amount_of_images = []
            total_rows = 0
            for index, count in out.items():
                if count < min_amount_of_pictures:
                    species_with_less_than_optimal_amount_of_images.append(index)
                    total_rows += 1

            total_rows = total_rows * min_amount_of_pictures
            print("Total number of extra rows: ", total_rows)
            world_df = world_df.loc[world_df['species'].isin(species_with_less_than_optimal_amount_of_images)]
            # loop that gets the species, which are below the required amount
            for item, count in less_than_list:
                world_specific = world_df.loc[world_df["species"] == item]
                world_specific = world_specific.iloc[:min_amount_of_pictures - count]

                df = pd.concat((df, world_specific), ignore_index=True)

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
        idx = df.index
        yolo_accepted = np.full(len(df.index), fill_value=np.nan).tolist()

        def save_img(row: pd.Series, index) -> Any:
            """ThreadPoolExecutor function, if file exists, returns the path. Tries to download, extract YOLO
            prediction, and if this returns a found butterfly, crop, resizes and saves.
            Returns the saved path.
            @param row: with download link
            @param index: base name of file
            @return: saved path of .npy file, and yolo_result
            """
            path = self.img_path_from_row(row, index)
            out = np.nan
            accepted = np.nan
            if 'path' in df.columns:
                if row['path'] in ERR_VALUES:
                    return row['path'], False

            if not os.path.exists(path):
                try:
                    res = requests.get(row[col], stream=True, timeout=40, verify=False).raw
                    img = Image.open(res)
                    if self.crop == 1:
                        model = YOLO('yolo/medium250e.pt')
                        res = obj_det(img, model, conf=0.50)
                        xywhn = res[0].boxes.xywhn
                        if xywhn.numel() > 0:
                            img = yolo_crop(img, xywhn)
                            accepted = True
                        else:
                            accepted = False
                    img = img.resize((constants['IMG_SIZE'], constants['IMG_SIZE']))
                    img = img.convert("RGB")
                    img = np.asarray(img)
                    np.save(path, img, allow_pickle=True)
                    out = path
                    self.debug(f"{out}: {accepted}")
                except requests.exceptions.Timeout:
                    self.warning(f"Timeout occurred for index {index}")
                    return "TIMEOUT", False
                except requests.exceptions.RequestException as e:
                    self.warning(f"Error occurred: {e} at index {index}")
                    return "REQUESTEXCEPTION", False
                except urllib3.exceptions.ReadTimeoutError:
                    self.warning(f"Read timed out on {index}")
                    return "TIMEOUT", False
                except ValueError:
                    self.error(f"Image name not applicable: {path}", stack_info=True, exc_info=True)
                    self.info(f"{out}: {accepted}")
                    return "ERROR", False
                except OSError:
                    self.error(f"Could not save file: {path}", stack_info=True, exc_info=True)
                    self.info(f"{out}: {accepted}")
                    return "ERROR", False
                except Exception:
                    self.error("Unknown error:", stack_info=True, exc_info=True)
                    self.info(f"{out}: {accepted}")
                    return "ERROR", False
            else:
                try:
                    out = row['path']
                    if pd.isna(out):
                        raise KeyError
                except KeyError:
                    out = path
                if self.crop == 1:
                    try:
                        accepted_tmp = row['yolo_accepted']
                        if pd.isna(accepted_tmp):
                            raise KeyError
                        else:
                            accepted = accepted_tmp
                    except KeyError:
                        model = YOLO('yolo/medium250e.pt')
                        res = obj_det(Image.fromarray(np.load(path, allow_pickle=True)), model, conf=0.25,
                                      img_size=(640, 640))
                        xywhn = res[0].boxes.xywhn
                        if xywhn.numel() > 0:
                            accepted = True
                        else:
                            accepted = False

                        self.debug(f"inherit {out}: {accepted}")
                    except TypeError:
                        accepted = False
            self.debug(f'{out}: {accepted}')
            return out, accepted

        with tqdm(total=len(df), smoothing=0.02) as pbar:
            with ThreadPoolExecutor(40) as executor:
                """Iterates through all rows, starts a thread to download, yolo-predict, save etc. each individual row, 
                then collects all results in a list, and insert these as columns 'path', 'yolo_accepted'."""
                futures = [executor.submit(save_img, row, index) for index, row in df.iterrows()]
                for _ in concurrent.futures.as_completed(futures):
                    pbar.update(1)
                concurrent.futures.wait(futures, timeout=None, return_when=ALL_COMPLETED)
                for i, ft in enumerate(futures):
                    paths[i], yolo_accepted[i] = ft.result()

        df['path'] = paths
        df['yolo_accepted'] = yolo_accepted
        df = self.ft_extractor.create_augmented_df(df=df, degrees=self.degrees)
        df.set_index(idx, inplace=True)
        return df

    def _csv_fits_self(self) -> bool:
        """Checks if .csv file exists returns if num images/species fits with the read file.
        @return: True if compliant, otherwise False
        """
        if os.path.exists(self.dataset_csv_filename):
            db = pd.read_csv(self.dataset_csv_filename, low_memory=False)
            df = self._get_working_df(db)
            try:
                if df['path'].isnull().values.any():
                    self.info(f"nulls in path: {len(df['path'].isnull())}")
                    return False
                if self.crop == 1 and df['yolo_accepted'].isnull().values.any():
                    self.info(f"nulls in yolo: {len(df['yolo_accepted'].isnull())}")
                    return False
                if self._num_rows and len(df) == self.num_rows:
                    return True
            except KeyError:
                return False
        return False

    def _partial_df(self) -> bool:
        """Checks if .csv file exists
        @return: True if compliant, otherwise False
        """
        return os.path.exists(self.dataset_csv_filename)

    def _merge_dfs_on_gbif(self, df: pd.DataFrame, df_label: pd.DataFrame) -> pd.DataFrame:
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

    def _get_working_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get current working window in db based on num species/images
        @param df: dataframe
        @return working window of db"""
        self.info(f"found {len(df['species'].unique())} unique species")
        if self.num_images and self.num_species:
            species = df['species'].unique()[:self.num_species]
            df = df.loc[df['species'].isin(species)].groupby('species').head(self.num_images)
        else:
            df = df.loc[df['species']].groupby('species').head(self.num_images)
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
        if constants['CROPPED'] == 1:
            df = df.loc[df['yolo_accepted'].isin(['True', True])]
            self.info(f"{len(df)} yolo_accepted")
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
            return self.num_images * self.num_species * 8
        elif self.degrees == "flip":
            return self.num_images * self.num_species * 2
        elif self.degrees == "rotate":
            return self.num_images * self.num_species * 4
        return self.num_images * self.num_species
