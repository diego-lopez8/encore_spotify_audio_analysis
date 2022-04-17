import pandas as pd
import numpy as np
import os
# same geneva scale from playlist_generator.py
# except we added a "_" for in love, as this was added at export time
geneva_scale = {"Wonder" : ["Happy", "Dazzling", "Alluring"], "Transcendence": ["Inspiring", "Spiritual"], "Tenderness":
        ["In_love", 'Sensual'], "Nostalgia": ["Nostalgic", "Sentimental", "Dreamy"], "Peacefulness": ["Peaceful", "Calm", "Relaxing"],
    "Power" : ["Energetic", "Fiery", "Heroic"], "Joyful" : ["Joyful", "Dancing"], "Tension" : ["Agitated", "Nervous"], "Sadness" : ["Sad", "Sorrow"]
}
# path to directory of csvs
directory = os.listdir("../data/audio_features")
csvs      = [elem for elem in directory if "features" in elem]
dfs       = []
# bring csvs into scope
for csv in csvs:
    print(csv)
    # create sub-emotions dfs from csvs
    vars()[f"{csv[:-4]}_df"] = pd.read_csv(f"../data/audio_features/{csv}")
    dfs.append(vars()[f"{csv[:-4]}_df"])
# add the sub-emotion of the df as a column to each row (for concatenation to the 9 emotion model later)
for i in range(len(dfs)):
    dfs[i]["sub-emotion"] = csvs[i][9:-4]   

# create the 9 geneva emotional emotion scale dataframes as a concatenation of the sub-emotions
for key, values in geneva_scale.items():
    vars()[f'{key}_df'] = pd.DataFrame()
    for value in values:
        vars()[f'{key}_df'] = pd.concat([vars()[f"{key}_df"], vars()[f"features_{value}_df"]], ignore_index=True)
    vars()[f'{key}_df']["emotion"] = key       
    vars()[f'{key}_df'].drop(columns="Unnamed: 0", axis=1, inplace=True)
    vars()[f'{key}_df'].to_csv(f"../data/processed/{key}.csv")
complete_dataset = pd.DataFrame()
for key in geneva_scale.keys():
    complete_dataset = pd.concat([complete_dataset, vars()[f"{key}_df"]], ignore_index=True)
#complete_dataset.drop(columns="Unnamed: 0", axis=1, inplace=True)
complete_dataset.to_csv("../data/processed/complete_dataset.csv")
complete_dataset.info()
