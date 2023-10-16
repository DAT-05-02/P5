from core.data.fetch import fetch_images, setup_dataset
from core.util.constants import RAW_DATA_PATH, RAW_LABEL_PATH, DATASET_PATH, LABEL_DATASET_PATH
from data.feature import FeatureExtractor

if __name__ == "__main__":
    df = setup_dataset(raw_dataset_path=RAW_DATA_PATH,
                       raw_label_path=RAW_LABEL_PATH,
                       label_dataset_path=LABEL_DATASET_PATH,
                       dataset_csv_filename=DATASET_PATH,
                       num_rows=50,
                       bfly=True)
    fetch_images(df, "identifier")
    ft_extractor = FeatureExtractor()
    df = ft_extractor.pre_process(df, "lbp", radius=7)
