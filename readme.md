# In This Moment Song Popularity Analysis, 2021 version

This project aims to identify the most important musical aspects of a song by ITM with regards to the song popularity.

## Prequisities

- python 3.7.1 or higher but at most 3.9
- poetry package for python (install via `python -m pip install poetry` if you do not have it yet)
- install the python package dependencies via `poetry install`, this also automatically creates a virtual environment for the project

## Usage

### Obtaining song data

- make a copy of `.env.example`, rename it to `.env` and add your client id and client secret
- run the function `get_and_save_songs()` from `src.processes`
- to obtain the data set this report uses, run `get_and_save_ITM_songs` from `src.processes` from terminal with command `poetry run python -m src.processes`

### Profiling the raw data

- run the function `src.profiling:profiling` which takes three attributes:
  - csv_source (str): the csv input file path to profile
  - report_title (str): the requested title of the report
  - profile_out (str): the html output file path
- to obtain the profile of the data set this report uses, simply run `poetry run python -m src.profiling`
