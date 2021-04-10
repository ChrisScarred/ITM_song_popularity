"""This file contains the class DataGetter used to obtain the songs data from Spotify API."""

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from src.models import *

class DataGetter:
	"""A class to obtain song data from Spotify API.

	Methods:
		get_albums(artist_uri): obtains all albums of an artist identified by artist_uri
		get_songs(album_id): obtains all songs of an album identified by album_id
		get_songs_data(artist_uri, non_inc_albums, non_inc_songs): obtains the song data for all songs of an artist identified by artist_uri except for the songs present in one of the non_inc_albums or being in non_inc_songs
		get_song_data(self, name, song_id): obtains the song data for a song identified by the name name and the id song_id
	"""

	def __init__(self, client_id: str, client_secret: str):
		"""Initialises a DataGetter object.

		Args:
			client_id (str): The client id as supplied by Spotify.
			client_secret (str): The client secret as supplied by Spotify.
		"""

		self.spotify = spotipy.Spotify(
			client_credentials_manager=SpotifyClientCredentials(
				client_id=client_id, 
				client_secret=client_secret
			)
		)

	def get_albums(self, artist_uri: str) -> List[Dict]:
		"""Obtains all albums of an artist identified by artist_uri.

		Args:
			artist_uri (str): the uri identifying the artist.

		Returns:
			List[Dict]: the list of albums in the form of json dicts.
		"""

		album_results = self.spotify.artist_albums(
			artist_uri, 
			album_type='album'
		)
		albums = album_results['items']
	
		while album_results['next']:
			album_results = self.spotify.next(album_results)
			albums.extend(album_results['items'])
		
		return albums

	def get_songs(self, album_id: str) -> List[Dict]:
		"""Obtains all songs of an album identified by album_id.

		Args:
			album_id (str): the album id.

		Returns:
			List[Dict]: the list of songs in the form of json dicts.
		"""

		song_results = self.spotify.album_tracks(album_id)
		songs = song_results['items']

		while song_results['next']:
			song_results = self.spotify.next(song_results)
			songs.extend(song_results['items'])

		return songs

	def get_songs_data(self, artist_uri: str, non_inc_albums: List[str] = [], non_inc_songs: List[str] = []) -> List[Song]:
		"""Obtains the song data for all songs of an artist.

		Identifies the authoring artist by the artist_uri as obtained from Spotify. Does not include songs present in albums whose name matches one of the non_inc_albums entries. Does not include songs whose name matches one of the non_inc_songs entries.

		Args:
			artist_uri (str): the artist uri as obtained from Spotify.
			non_inc_albums (List[str], optional): The list of names of albums to exclude. Defaults to [].
			non_inc_songs (List[str], optional): The list of names of songs to exclude. Defaults to [].

		Returns:
			List[Song]: the songs data as a list of Song objects.
		"""

		songs_data = []
		albums = self.get_albums(artist_uri)
		
		for album in albums:
			album_name = album['name']
			album_id = album['id']

			if not album_name in non_inc_albums:
				print("Processing album `%s` with id `%s`" % (album_name, album_id))			

				songs = self.get_songs(album_id)

				for song in songs:
					song_id = song['id']
					song_name = song['name']

					if not song_name in non_inc_songs:
						print('- Processing song `%s` with id `%s`' % (song_name, song_id))

						songs_data.append(self.get_song_data(song_name, song_id))
					
					else:
						print('- Skipping song `%s` with id `%s`' % (song_name, song_id))
			
			else:
				print("Skipping album `%s` with id `%s`" % (album_name, album_id))

		return songs_data

	def get_song_data(self, name: str, song_id: str) -> Song:
		"""Obtains the song data for a song identified by the name string name and the id song_id.

		Args:
			name (str): the song name.
			song_id (str): the song id.

		Returns:
			Song: the song data as an object of type Song.
		"""

		features = self.spotify.audio_features(tracks=[song_id])[0]
		song_data = self.spotify.track(song_id)
		analysis = self.spotify.audio_analysis(song_id)

		return Song(
			name = name,
			id = song_id,
			duration_ms = features["duration_ms"],
			key = features["key"],
			mode = features["mode"],
			time_signature = features["time_signature"],
			acousticness = features["acousticness"],
			danceability = features["danceability"],
			energy = features["energy"],
			instrumentalness = features["instrumentalness"],
			loudness = features["loudness"],
			speechiness = features["speechiness"],
			valence = features["valence"],
			tempo = features["tempo"],
			explicit = 1 if song_data.get("explicit") else 0,
			track_number = song_data["track_number"],
			complexity = len(analysis["segments"]),
			popularity = song_data["popularity"]
		)
