"""
Author: Diego Lopez (diego.lopez@nyu.edu)
Date: 10/25/2021
This file contains the code needed to generate the complete tracklists from the playlist csvs.
This file takes a csv of playlists and outputs a csv of the complete concatenated tracklists of these playlists
This requires a first run of playlist_csv_generator.py 
Please increase your request timeout in home/.local/lib/python3.xx/site-packages/spotipy/client.py as
some of the playlists are very large
"""
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import json
import pandas as pd 
import numpy as np 
import os
load_dotenv()
# spotipy and spotify authentication 
client_id     = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
redirect_uri  = os.getenv("REDIRECT_URI")
scope         = 'user-top-read'
sp            = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope, client_id=client_id, 
        client_secret=client_secret, redirect_uri=redirect_uri))

def main():
    # get csvs inside current directory
    # may need to change path in listdir() if different directory
    directory = os.listdir()
    csvs      = [elem for elem in directory if ".csv" in elem]
    counter   = 0
    # skips updates of current lists (I got lost of timeouts when running script due to high volumne of querries)
    while counter < len(csvs):
        if csvs[counter][:6] == "tracks":
            strtodel   = csvs[counter][7:]
            subcounter = 0
            while subcounter < len(csvs):
                if strtodel in csvs[subcounter]:
                    del csvs[subcounter]
                    subcounter -= 1
                    counter    -= 1
                subcounter += 1
        counter += 1

    for csv in csvs:
        print ("Queued to generate tracks for", csv)
    for csv in csvs:
        print("generating features and tracks for  ", csv)
        playlist_df              = pd.read_csv(csv)
        total_tracklist          = []
        total_tracklist_features = []

        for i in range(len(playlist_df)):
            print("Generating tracks for playlist:", playlist_df.iloc[i,4])
            user        = playlist_df.iloc[i, -3]
            playlist_id = playlist_df.iloc[i, 4]
            # using our function to collect all tracks from playlist
            results     = get_playlist_tracks(user, playlist_id)
            # normalize the results to remove json format
            results_df  = pd.json_normalize(results)
            tids        = []
            # concatenate to the total of all playlists
            # here is where we would also include the playlist number it came from if sampling
            total_tracklist.extend(results)
            for elem in results_df["track.uri"]:
                tids.append(elem)
            nanDeleter(tids)
            # only call batch_audio_features on tids that are registered with spotify
            features = batch_audio_features(tids)
            total_tracklist_features.extend(features)

        total_tracklist_df          = pd.json_normalize(total_tracklist)
        total_tracklist_features_df = pd.json_normalize(total_tracklist_features)
        # exporting results to csvs
        total_tracklist_df.to_csv(f"../data/metadata/tracks_{csv}")
        total_tracklist_features_df.to_csv(f"../data/audio_features/features_{csv}")

def nanDeleter(lst):
    # deletes nan values from given list. 
    # needed because we want to only pass calls to audio_features api if the song has a TID 
    # if song has TID then spotify has it on cloud, ie filtering for local songs
    i = 0
    while i < len(lst):
        if lst[i] is np.nan:
            lst.pop(i)
            i -= 1
        i += 1

def get_playlist_tracks(username,playlist_id):
    # get tracks from a playlist and return them
    # spotipy limits the user_playlist_tracks function to return 100 at a time
    # use the next() function to get complete playlist
    results = sp.user_playlist_tracks(username,playlist_id)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return tracks

def batch_audio_features(id_lst):
    # spotipy audio_features api call has a maximum of 100 per call
    # this function takes a list with length >100 and returns all audio features
    if len(id_lst) <= 100: 
        features = sp.audio_features(id_lst)
        return features
    else:
        id_len   = len(id_lst)
        curr_pos = 0
        features = []
        while id_len > 100:
            feature_page = sp.audio_features(id_lst[curr_pos: curr_pos+100])
            features.extend(feature_page)
            curr_pos += 100
            id_len   -= 100
        if id_len > 0:
            feature_page = sp.audio_features(id_lst[curr_pos: curr_pos + id_len])
            features.extend(feature_page)
        return features

if __name__=="__main__":
   main()