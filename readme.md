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
- to obtain the data set this report uses, run `get_and_save_ITM_songs` from `src.processes` from terminal with command `python -m src.processes`
