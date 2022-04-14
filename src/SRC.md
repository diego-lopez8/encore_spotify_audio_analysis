# SRC.md

By: Diego Lopez (diego.lopez@nyu.edu)

## How to run this code

Please head to https://developer.spotify.com/ to register with Spotify directly. This is required to use the Spotify API.

After registering and creating a project, use the Client ID, Client Secret, and Redirect URI to properly run Spotipy calls. 

Spotipy's documentation can be found here: https://spotipy.readthedocs.io/en/2.19.0/

Please also note when running tracklist_generator.py, some playlists are very large and may cause your script to time out.

To avoid this, I've added dynamic restarting controls so if the script exit, the next run will only download missing files.

Please increase the time out by heading to Spotipy's Client.py file and increasing it directly. I was able to run the script

without any interruptions with a time out of 40 seconds. 