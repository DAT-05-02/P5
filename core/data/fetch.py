import concurrent
from concurrent.futures import ThreadPoolExecutor, ALL_COMPLETED
from typing import Any

import numpy as np
import requests
import pandas as pd
import os
from PIL import Image
from core.util.util import timing
from core.util.constants import IMGDIR_PATH, MERGE_COLS, BFLY_FAMILY, BFLY_LIFESTAGE, DATASET_PATH, DIRNAME_DELIM
from data.feature import FeatureExtractor


def setup_dataset(raw_dataset_path: str,
                  raw_label_path: str,
                  label_dataset_path: str,
                  dataset_csv_filename: str,
                  num_rows=None,
                  sort=False,
                  bfly: list = []):
    """ Loads a file, converts to csv if none exists, or loads an existing csv into a pd.DateFrame object
    @param raw_label_path: path to original label dataset file
    @param raw_dataset_path: path to original dataset file
    @param label_dataset_path: path to label csv file
    @param dataset_csv_filename: filename for the csv file
    @param num_rows: number of rows to include
    @param sort: if dataset should be sorted by species
    @param bfly: list of species that is included in dataset, have "all" in list for only butterflies (no moths)
    Returns: pandas.DataFrame object with data

    """
    if not os.path.exists(IMGDIR_PATH):
        os.makedirs(IMGDIR_PATH)
    if os.path.exists(dataset_csv_filename):
        df = pd.read_csv(dataset_csv_filename, low_memory=False)
        if num_rows and len(df) == num_rows:
            return df

    df: pd.DataFrame = pd.read_csv(raw_dataset_path, sep="	", low_memory=False)

    if os.path.exists(label_dataset_path):
        df_label: pd.DataFrame = pd.read_csv(label_dataset_path, low_memory=False)
    else:
        df_label: pd.DataFrame = pd.read_csv(raw_label_path, sep="	", low_memory=False)
        df_label.to_csv(label_dataset_path, index=False)
    print(df[df.columns])
    drop_cols([df, df_label], MERGE_COLS)
    df = df.merge(df_label[df_label['gbifID'].isin(df['gbifID'])], on=['gbifID'])
    df = df.loc[~df['lifeStage'].isin(BFLY_LIFESTAGE)]
    df = df.dropna(subset=['species'])
    if bfly:
        if "all" in bfly:
            df = df.loc[df['family'].isin(BFLY_FAMILY)]
        else:
            df = df.loc[df['species'].isin(bfly)]
    print(df.shape)
    if sort:
        df.sort_values(by=['species'], inplace=True)
    print(f"found {len(df['species'].unique())} unique species")
    if num_rows:
        df.drop(df.index[num_rows:], inplace=True)
    df.reset_index(inplace=True, drop=True)
    df.to_csv(dataset_csv_filename, index=False)
    return df

def pad_dataset(df, raw_dataset_path: str, raw_label_path: str, csv_path: str, min_amount_of_pictures=3):
    run_correction = False

    for count in df['species'].value_counts():
        if min_amount_of_pictures > count:
            run_correction = True


    '''
    for folder in df["species"].unique:
        if len(os.listdir(IMGDIR_PATH + folder)) < min_amount_of_pictures:
            run_correction = True
    '''
    if run_correction:
        # just need to make it so that if there are species in the dataset, which are below the min_amount, then find them in the global dataset and put into the other one
        world_df: pd.DataFrame = pd.read_csv(raw_dataset_path, sep="	", low_memory=False)
        # This file is separated by commas
        world_df_labels: pd.DataFrame = pd.read_csv(raw_label_path, sep=",", low_memory=False)

        drop_cols([world_df, world_df_labels], MERGE_COLS)

        world_df = world_df.merge(world_df_labels[world_df_labels['gbifID'].isin(world_df['gbifID'])], on=['gbifID'])

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

        world_df = world_df.iloc[:totalRows]

        df = pd.concat([df, world_df], ignore_index=True, sort=False)

    return df
def drop_cols(dfs, cols):
    for df in list(dfs):
        df.drop(columns=[col for col in df if col not in MERGE_COLS], inplace=True)


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
    return out + f"_0.{extension}"


@timing
def fetch_images(df: pd.DataFrame, col: str):
    """
    Fetches all image links in a DataFrame column to path defined by :func:`~fetch.img_path_from_row`
    and assigns the column value to the path saved to.
    @param df: the DataFrame containing links
    @param col: which column the links are in
    """
    paths = np.full(len(df.index), fill_value=np.nan).tolist()

    def save_img(row: pd.Series, index) -> Any:
        path = img_path_from_row(row, index)
        out = np.nan
        if not os.path.exists(path):
            try:
                img = Image.open(requests.get(row[col], stream=True, timeout=40).raw)
                img = FeatureExtractor.make_square_with_bb(img)
                img = img.resize((416, 416))
                img.save(path)
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
        #df = df[df[['path']].notnull().any(axis=1)]
        df.to_csv(DATASET_PATH, index=False)
