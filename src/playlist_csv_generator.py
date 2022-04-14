# playlist_csv_generator.py
"""
Author: Diego Lopez (diego.lopez@nyu.edu)
Date: 10/23/2021
This file contains the code to generate CSV files of the various playlists associated with the appropriate terms
This file should be considered a pre-requisite to the file that generates the tracklist from each playlist and the associated features of those tracks
"""

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import json
import pandas as pd 
import os
import numpy as np 
load_dotenv()
# spotipy and spotify authentication 
client_id     = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
redirect_uri  = os.getenv("REDIRECT_URI")
scope         = 'user-top-read'
sp            = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope, client_id=client_id, 
                    client_secret=client_secret, redirect_uri=redirect_uri))

def main():
    # creating the geneva scale in python of the highest correlation submoods and most "operationalize-able"
    # some have been left out because there was not much data as of my preliminary scans using the GUI search 
    # TODO: automate possible sample scanning for all submoods of GEMS
    geneva_scale = {"Wonder" : ["Happy", "Dazzling", "Alluring"], "Transcendence": ["Inspiring", "Spiritual"], "Tenderness":
         ["In love", 'Sensual'], "Nostalgia": ["Nostalgic", "Sentimental", "Dreamy"], "Peacefulness": ["Peaceful", "Calm", "Relaxing"],
        "Power" : ["Energetic", "Fiery", "Heroic"], "Joyful" : ["Joyful", "Dancing"], "Tension" : ["Agitated", "Nervous"], "Sadness" : ["Sad", "Sorrow"]
    }
    for mood in geneva_scale.keys():
        limit = 300//len(geneva_scale[mood])
        for submood in geneva_scale[mood]:
            # we want 300 playlists for each 
            # generate playlists datasets
            results_full = pd.DataFrame()
            offset=0
            while 25*offset < limit:
                results           = sp.search(submood, type="playlist", limit=25, offset=25*offset+1)
                results_normed_df = pd.json_normalize(results["playlists"]['items'])
            #results_offset_50 = sp.search(submood, type="playlist", limit=limit, offset=limit*offset)
            #results_page2_df  = pd.json_normalize(results_offset_50["playlists"]['items'])
                results_full      = pd.concat([results_full, results_normed_df], ignore_index=True) 
                offset+=1
            submood = submood.replace(" ", "_")
            # exporting to csvs
            results_full.to_csv(f"../data/playlists/{submood}.csv")
if __name__=="__main__":
   main()