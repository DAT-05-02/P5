import pandas as pd


# Converts the world data to have less useless rows
def world_converter():
    df = pd.DataFrame = pd.read_csv("leopidotera-world/occurrence.txt", sep="	", low_memory=False)
    # df.to_csv("occurrence.csv", index=False)
    print(df[df.columns])
    df.drop(columns=[col for col in df if
                     col not in ['genericName', 'species', 'family', 'stateProvince', 'gbifID', 'identifier',
                                 'iucnRedListCategory', 'lifeStage']], inplace=True)
    print(df[df.columns])
    df.to_csv("leopidotera-world/occurrence.csv", index=False)
    with open("leopidotera-world/occurrence.txt", 'w') as f:
        dfAsString = df.to_string(header=True, index=False)
        f.write(dfAsString)