"""This file contains the default processes supported by this module.
"""

import os
from dotenv import dotenv_values

from src.data_getter import DataGetter
from src.utils import *
from src.models import *


def get_and_save_songs(artist_uri: str, json_name: str, csv_name: str, non_inc_albs: List[str] = [], non_inc_songs: List[str] = []):
    """Obtains songs from a specific artist and saves them as .json and .csv.

    Identifies the authoring artist by the artist_uri as obtained from Spotify. Does not include songs present in albums whose name matches one of the non_inc_albums entries. Does not include songs whose name matches one of the non_inc_songs entries. Saves the songs as .json into the file with the name json_name. Saves the songs as .csv into the file with the name csv_name. 

    Args:
        artist_uri (str): the artist uri as obtained from Spotify.
        json_name (str): the file name specifying where to store the .json format of songs data.
        csv_name (str): the file name specifying where to store the .csv format of songs data.
        non_inc_albs (List[str], optional): The list of names of albums to exclude. Defaults to [].
        non_inc_songs (List[str], optional):  The list of names of songs to exclude. Defaults to [].
    """

    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    config = dotenv_values(os.path.join(base_dir, ".env"))
    client_id = config.get("SPOTIPY_CLIENT_ID")
    client_secret = config.get("SPOTIPY_CLIENT_SECRET")

    data_getter = DataGetter(client_id, client_secret)
    songs = data_getter.get_songs_data(artist_uri, non_inc_albs, non_inc_songs)

    json_loc = os.path.join(base_dir, json_name)
    success = save_songs_as_json(songs, json_loc)

    if success:
        print("Songs data saved into the file `%s`." % json_loc)

        csv_loc = os.path.join(base_dir, csv_name)
        success = json_to_csv(json_loc, csv_loc)

        if success:
            print("Songs data transformed from `%s` to `%s`." % (json_loc, csv_loc))
        
        else:
            print("Songs data failed to transform into the .csv fromat.")

    else:
        print("Saving was not successful.")


def get_and_save_ITM_songs():
    """Sets up all parameters as needed to obtain all studio songs by In This Moment. 
    """
    get_and_save_songs("spotify:artist:6tbLPxj1uQ6vsRQZI2YFCT", "database/itm_songs.json", "database/itm_songs.csv", non_inc_albs=["Blood at the Orpheum (Live)", "Blood"], non_inc_songs=["Interview (Bonus)"])


if __name__=="__main__":
    get_and_save_ITM_songs()
