"""Preprocessing functions
"""

from src.models import *
import pandas as pd
import re
from datetime import datetime


def preprocess(raw_path: str, preprocessed_path: str) -> None:
    """Preprocesses the song information from the raw csv file.

    Args:
        raw_path (str): the path to the raw csv file.
        preprocessed_path (str): the path to the csv file where the preprocessed data are stored.
    """
    raw_data = pd.read_csv(raw_path)
    processed = pd.DataFrame({
        "name": raw_data["name"],
        "name_len": [len(re.sub("\s\(feat.+\)","",n)) for n in raw_data["name"]],
        "track_number": raw_data["track_number"],
        "duration": normalise(raw_data["duration_ms"]),
        "key": [KEYS.get(num, "") for num in raw_data["key"]],
        "mode": [MODES.get(num, "") for num in raw_data["mode"]],
        "time_signature": raw_data["time_signature"],
        "acousticness": raw_data["acousticness"],
        "danceability": raw_data["danceability"],
        "energy": raw_data["energy"],
        "instrumentalness": raw_data["instrumentalness"],
        "loudness": normalise(raw_data["loudness"]),
        "speechiness": raw_data["speechiness"],
        "valence": raw_data["valence"],
        "tempo": normalise(raw_data["tempo"]),
        "explicit": [bool(i) for i in raw_data["explicit"]],
        "complexity": normalise(raw_data["complexity"]),
        "popularity_abs": raw_data["popularity"],
        "popularity_norm": normalise(raw_data["popularity"]),
        "age_days": get_age_days(raw_data["release_date"]),
    })
    processed.to_csv(preprocessed_path)


def get_age_days(data: List) -> List:
    """Converts a list of dates in strings into the list of age in days.
    """
    date_format = "%Y-%m-%d"
    today = datetime.now()
    return [(today-datetime.strptime(date, date_format)).days for date in data]


def normalise(data: List) -> List:
    """Performs min-max normalisation (also called standardisation).
    """
    min_val = min(data)
    diff = max(data) - min_val
    return [((x-min_val)/diff) for x in data]


if __name__=="__main__":
    preprocess("database/itm_songs.csv", "database/itm_songs_preprocessed.csv")
