import os
import sys

from core.data.fetch import fetch_images, setup_dataset, pad_dataset
from core.util.constants import RAW_DATA_PATH, RAW_LABEL_PATH, DATASET_PATH, LABEL_DATASET_PATH, RAW_WORLD_DATA_PATH, RAW_WORLD_LABEL_PATH


if __name__ == "__main__":
    sys.path.append(f"{os.getcwd()}{os.sep}core")
    if not os.getcwd().split(os.sep)[-1] == "core":
        os.chdir("core")
    df = setup_dataset(raw_dataset_path=RAW_DATA_PATH,
                       raw_label_path=RAW_LABEL_PATH,
                       label_dataset_path=LABEL_DATASET_PATH,
                       dataset_csv_filename=DATASET_PATH,
                       num_rows=50,
                       bfly=["all"])
    df = pad_dataset(df, RAW_WORLD_DATA_PATH, RAW_WORLD_LABEL_PATH, DATASET_PATH, min_amount_of_pictures=5)
    fetch_images(df, "identifier")
    '''
    ft_extractor = FeatureExtractor(log_level=logging.INFO)
    df = ft_extractor.pre_process(df, "lbp", radius=7, should_bb=True, should_resize=True)
    model = Model(df)
    # model.load()
    model.print_dataset_info()
    model.compile()
    model.split_dataset()
    model.fit(5)  # Epochs
    model.save()
    model.evaluate_and_show_predictions()
    '''
