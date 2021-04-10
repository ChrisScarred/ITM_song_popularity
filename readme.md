# In This Moment Song Popularity Analysis, 2021 version

This project aims to identify the most important musical aspects of a song by ITM with regards to the song popularity.

This version is currently in development.

## Prequisities

- make and activate a python virtual environment
- install the dependencies via `python -m pip install -r requirements.txt`

## Usage

### Obtaining song data

- make a copy of `.env.example`, rename it to `.env` and add your client id and client secret
- run the function `get_and_save_songs()` from `src.processes`
- to obtain the data set this report uses, following script was used:

```python
from src.processes import get_and_save_songs

def main():
    artist_uri = "spotify:artist:6tbLPxj1uQ6vsRQZI2YFCT"
    non_inc_albs = ["Blood at the Orpheum (Live)", "Blood"]
    non_inc_tracks = ["Interview (Bonus)"]
    json_name = "data/itm_songs.json"
    csv_name = "data/itm_songs.csv"

    get_and_save_songs(artist_uri, json_name, csv_name, non_inc_albs, non_inc_tracks)

if __name__ == "__main__":
    main()
```
