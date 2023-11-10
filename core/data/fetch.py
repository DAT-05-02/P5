import concurrent
import logging
from concurrent.futures import ThreadPoolExecutor, ALL_COMPLETED
from typing import Any

import numpy as np
import requests
import pandas as pd
import os
from PIL import Image

from core.util.logging.logable import Logable
from core.util.util import timing, setup_log, log_ent_exit
from core.util.constants import IMGDIR_PATH, MERGE_COLS, BFLY_FAMILY, BFLY_LIFESTAGE, DATASET_PATH, DIRNAME_DELIM, RAW_WORLD_DATA_PATH, RAW_WORLD_LABEL_PATH
from core.data.feature import FeatureExtractor


class Database(Logable):
    def __init__(self,
                 min_amount_of_pictures,
                 raw_dataset_path: str,
                 raw_label_path: str,
                 label_dataset_path: str,
                 dataset_csv_filename: str,
                 bfly: list,
                 ft_extractor: FeatureExtractor,
                 degrees: str,
                 num_rows=None,
                 sort=False,
                 log_level=logging.INFO,
                 ):
        """ Loads a file, converts to csv if none exists, or loads an existing csv into a pd.DateFrame object
            @param raw_label_path: path to original label dataset file
            @param raw_dataset_path: path to original dataset file
            @param label_dataset_path: path to label csv file
            @param dataset_csv_filename: filename for the csv file
            @param num_rows: number of rows to include
            @param sort: if dataset should be sorted by species
            @param bfly: list of species that is included in dataset, have "all" in list for only butterflies (no moths)"""
        super().__init__()
        setup_log(log_level=log_level)
        self.min_amount_of_pictures = min_amount_of_pictures
        self.raw_dataset_path = raw_dataset_path
        self.raw_label_path = raw_label_path
        self.label_dataset_path = label_dataset_path
        self.dataset_csv_filename = dataset_csv_filename
        self.ft_extractor = ft_extractor
        self.bfly = bfly
        self.degrees = degrees
        self._num_rows = num_rows
        self.num_rows = self.num_rows()
        self.sort = sort

    @log_ent_exit
    def num_rows(self):
        if self._num_rows is None:
            return None
        elif self.degrees == "all":
            return self._num_rows * 8
        elif self.degrees == "flip":
            return self._num_rows * 2
        elif self.degrees == "rotate":
            return self._num_rows * 4
        return self._num_rows

    def setup_dataset(self):

        if not os.path.exists(IMGDIR_PATH):
            os.makedirs(IMGDIR_PATH)
        if os.path.exists(self.dataset_csv_filename):
            df = pd.read_csv(self.dataset_csv_filename, low_memory=False)
            self.log.debug(f"len(df): {len(df)}, self.num_rows: {self.num_rows}")
            if self._num_rows and len(df) == self.num_rows:
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
        if self._num_rows:
            df.drop(df.index[self._num_rows:], inplace=True)
        df.reset_index(inplace=True, drop=True)

        df = self.pad_dataset(df, RAW_WORLD_DATA_PATH, RAW_WORLD_LABEL_PATH, min_amount_of_pictures=self.min_amount_of_pictures)
        df = self.fetch_images(df, "identifier")
        df.reset_index(inplace=True, drop=True)
        df.to_csv(self.dataset_csv_filename, index=False)
        return df

    @staticmethod
    def img_path_from_row(row: pd.Series, index: int, column="identifier", extra=None):
        """Generates path for an image based on row and index with optional column, in which the image link is.
        @param row: Series to extract file path from
        @param index: of the row. Series objects don't inherently know which index they are in a DataFrame.
        @param column: (optional) which column the file path is in
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
        extension = row[column].split(".")[-1]
        if len(extension) < 1:
            extension = "jpg"
        out = f"{IMGDIR_PATH}{row[9].replace(' ', DIRNAME_DELIM)}/{index}"
        if extra:
            out += extra
        return out + "_0.npy"
    @staticmethod
    def pad_dataset(df, raw_dataset_path: str, raw_label_path: str, min_amount_of_pictures=3):
        run_correction = False

        print("coo coo", df)

        values = df['species'].value_counts().keys().tolist()
        counts = df['species'].value_counts().tolist()

        itt = 0

        less_than_list = []

        for count in counts:
            if min_amount_of_pictures > count:
                less_than_list.append((values[itt], count))
                run_correction = True
            itt += 1

        if run_correction:
            # just need to make it so that if there are species in the dataset, which are below the min_amount, then find them in the global dataset and put into the other one
            world_df: pd.DataFrame = pd.read_csv(raw_dataset_path, sep="	", low_memory=False)
            # This file is separated by commas
            world_df_labels: pd.DataFrame = pd.read_csv(raw_label_path, sep=",", low_memory=False)

            Database.drop_cols([world_df, world_df_labels])

            world_df = world_df.merge(world_df_labels[world_df_labels['gbifID'].isin(world_df['gbifID'])],
                                      on=['gbifID'])
            # print("world_df: ", world_df.columns)

            out = df["species"].value_counts()

            species_with_less_than_optimal_amount_of_images = []

            totalRows = 0
            for index, count in out.items():
                if count < min_amount_of_pictures:
                    species_with_less_than_optimal_amount_of_images.append(index)
                    totalRows += 1

            totalRows = totalRows * min_amount_of_pictures
            print("Total number of extra rows: ", totalRows)
            world_df = world_df.loc[world_df['species'].isin(species_with_less_than_optimal_amount_of_images)]
            # loop that gets the species, which are below the required amount
            for item, count in less_than_list:
                # print("World pictures to add: ", min_amount_of_pictures - count)
                world_specific = world_df.loc[world_df["species"] == item]
                world_specific = world_specific.iloc[:min_amount_of_pictures - count]
                df = df.append(world_specific, ignore_index=True)
            # df = Database.fetch_images(df, "identifier")
        return df

    @timing
    def fetch_images(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Fetches all image links in a DataFrame column to path defined by :func:`~fetch.img_path_from_row`
        and assigns the column value to the path saved to.
        @param df: the DataFrame containing links
        @param col: which column the links are in
        """
        paths = np.full(len(df.index), fill_value=np.nan).tolist()

        def save_img(row: pd.Series, index) -> Any:
            path = self.img_path_from_row(row, index)
            out = np.nan
            if not os.path.exists(path):
                try:
                    img = Image.open(requests.get(row[col], stream=True, timeout=40, verify=False).raw)
                    img = FeatureExtractor.make_square_with_bb(img)
                    img = img.resize((416, 416))
                    img = np.asarray(img)
                    np.save(path, img, allow_pickle=True)
                    out = path
                    print(path)
                except requests.exceptions.Timeout:
                    print(f"Timeout occurred for index {index}")
                except requests.exceptions.RequestException as e:
                    print(f"Error occurred: {e}")
                except ValueError as e:
                    print(f"Image name not applicable: {e}")
                except OSError as e:
                    print(f"Could not save file: {e}")
                except Exception as e:
                    print(f"Unknown error: {e}")
            else:
                out = path
            return out

        with ThreadPoolExecutor(100) as executor:
            futures = [executor.submit(save_img, row, index) for index, row in df.iterrows()]
            concurrent.futures.wait(futures, timeout=None, return_when=ALL_COMPLETED)
            for i, ft in enumerate(futures):
                paths[i] = ft.result()

            df['path'] = paths
            df.dropna(subset=['path'], inplace=True)
        df = self.ft_extractor.create_augmented_df(df=df, degrees=self.degrees)
        self.log.info(df)
        df.to_csv(DATASET_PATH, index=False)
        return df

    @staticmethod
    def drop_cols(dfs):
        for df in list(dfs):
            df.drop(columns=[col for col in df if col not in MERGE_COLS], inplace=True)
