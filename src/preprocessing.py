"""Preprocessing functions
"""

from src.models import *
import pandas as pd
import re
from datetime import datetime


KEYS = {
    0: "c_or_b_sharp",
    1: "c_sharp_or_d_flat",
    2: "d",
    3: "d_sharp",
    4: "e",
    5: "f",
    6: "f_sharp",
    7: "g",
    8: "g_sharp",
    9: "a",
    10: "b_flat",
    11: "b"
}


MODES = {
    0: "minor",
    1: "major"
}


def preprocess(raw_path: str, preprocessed_path: str) -> None:
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
    date_format = "%Y-%m-%d"
    today = datetime.now()
    return [(today-datetime.strptime(date, date_format)).days for date in data]


def normalise(data: List) -> List:
    min_val = min(data)
    diff = max(data) - min_val
    return [((x-min_val)/diff) for x in data]


if __name__=="__main__":
    preprocess("database/itm_songs.csv", "database/itm_songs_preprocessed.csv")
