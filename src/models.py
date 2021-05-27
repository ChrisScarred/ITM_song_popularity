"""This file contains the class Song which represents song data as obtained from Spotify API."""

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
