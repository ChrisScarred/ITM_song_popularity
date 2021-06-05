"""This file contains various utility functions.
"""

import pandas as pd
import json

from src.models import *


def save_songs_as_json(songs: List[Song], file_name: str) -> bool:
	"""Saves the list of songs in the Song or ProcessedSong object format as .json format.

	Args:
		songs (List[Song]): the list of songs data to save.
		file_name (str): the file name where the songs are stored.

	Returns:
		bool: successfulness of the saving action.
	"""
	songs = [song.dict() for song in songs]
	try:
		with open(file_name, "w+") as f:
			json.dump(songs, f)
	except:
		return False
	return True


def json_to_csv(input_file: str, output_file: str) -> bool:
	"""Transforms .json data into .csv data.

	Args:
		input_file (str): the file name for the .json file.
		output_file (str): the file name for the file where to store the .csv format.

	Returns:
		bool: The successfulness of the conversion.
	"""
	try:
		df = pd.read_json(path_or_buf=input_file)
		df.to_csv(output_file)
	except:
		return False
	return True
