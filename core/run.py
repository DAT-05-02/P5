from core.data.fetch import fetch_images, setup_dataset
DATA_PATH="leopidotera-dk.csv"
LABEL_PATH="leopidotera-dk/occurrence.txt"

if __name__ == "__main__":
    df = setup_dataset("leopidotera-dk/multimedia.txt", LABEL_PATH, DATA_PATH, 400)
    fetch_images(df, "identifier")