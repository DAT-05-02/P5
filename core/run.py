from core.data.fetch import fetch_images, setup_dataset
DATA_PATH="leopidotera-dk.csv"
LABEL_PATH="leopidotera-dk/occurrence.txt"

if __name__ == "__main__":
    #put specific species in and uncomment to filter dataset
    #chosenButterflies = ["Aglais io", "Aglais urticae"]
    df = setup_dataset("leopidotera-dk/multimedia.txt", LABEL_PATH, DATA_PATH, 50)
    #fetch_images(df, "identifier")
