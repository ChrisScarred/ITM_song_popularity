"""This file contains the class Song which represents song data as obtained from Spotify API.
"""

from pydantic import BaseModel
from typing import List, Dict


class Song(BaseModel):
    """Represents the raw data as received from the Spotify API.
    """
    name: str
    id: str
    track_number: int
    duration_ms: float
    key: int
    mode: int
    time_signature: int
    acousticness: float
    danceability: float
    energy: float
    instrumentalness: float
    loudness: float
    speechiness: float
    valence: float
    tempo: float
    explicit: int
    complexity: int
    popularity: int
    release_date: str

"""Represents the values of musical keys with regards to their name. 
"""
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

"""Represents the values of musical modes with regards to their name.
"""
MODES = {
    0: "minor",
    1: "major"
}
