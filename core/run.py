from core.data.fetch import fetch_images, setup_dataset
DATA_PATH="leopidotera-dk.csv"

if __name__ == "__main__":
    df = setup_dataset("leopidotera-dk/multimedia.txt", DATA_PATH)
    #
    # Uncomment below depending on how many images you want to download
    #
    df = df[10000:10101]
    fetch_images(df, "identifier")