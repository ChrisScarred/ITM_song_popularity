import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

SPOTIPY_CLIENT_ID='17fc2297bbdf4be3b339cf17f8307c45'
SPOTIPY_CLIENT_SECRET='46eaf0adb7c8472994ed2404c7700142'
SPOTIPY_REDIRECT_URI='http://localhost:8888/callback'


class DataGetter:
	def __init__(self, artist_uri, non_inc_albs, non_inc_tracks, name):
		self.artist_uri = artist_uri
		self.non_inc_tracks = non_inc_tracks
		self.non_inc_albs = non_inc_albs
		self.spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET))
		self.name = name

	def getData(self):
		album_results = self.spotify.artist_albums(self.artist_uri, album_type='album')
		albums = album_results['items']

		songs_data = pd.DataFrame({
			"name":[], 
			"id":[],
			"track_number":[],
			"duration_ms":[],
			"key":[],
			"mode":[],
			"time_signature":[],
			"acousticness":[],
			"danceability":[],
			"energy":[],
			"instrumentalness":[],
			"loudness":[],
			"speechiness":[],
			"valence":[],
			"tempo":[],
			"explicit":[],
			"complexity":[],
			"popularity":[]
			})
	
		while album_results['next']:
			album_results = self.spotify.next(album_results)
			albums.extend(album_results['items'])

		for album in albums:
			album_name = album['name']
			if all([album_name != x for x in self.non_inc_albs]): 
				album_id = album['id']

				print("Processing album "+album_name)			

				track_results = self.spotify.album_tracks(album_id)
				tracks = track_results['items']

				while track_results['next']:
					track_results = self.spotify.next(track_results)
					tracks.extend(track_results['items'])

				for track in tracks:
					track_id = track['id']
					track_name = track['name']

					if all(track_name != x for x in self.non_inc_tracks):
						print('- - Processing track '+track_name)
						songs_data = self.getSongData(songs_data, track_name, track_id)
		songs_data.to_csv(self.name+".csv")			
		print("Done. Results saved to file "+self.name+".csv")
		return songs_data

	def getSongData(self, df, name, track_id):
		features = self.spotify.audio_features(tracks=[track_id])[0]
		track_data = self.spotify.track(track_id)
		analysis = self.spotify.audio_analysis(track_id)

		duration_ms = features["duration_ms"]
		key = features["key"]
		mode = features["mode"]
		time_signature = features["time_signature"]
		acousticness = features["acousticness"]
		danceability = features["danceability"]
		energy = features["energy"]
		instrumentalness = features["instrumentalness"]
		loudness = features["loudness"]
		speechiness = features["speechiness"]
		valence = features["valence"]
		tempo = features["tempo"]
		explicit = 1 if track_data["explicit"] else 0
		track_number = track_data["track_number"]
		complexity = len(analysis["segments"])

		popularity = track_data["popularity"]


		song_data = pd.DataFrame({
			"name":[name], 
			"id":[track_id],
			"track_number":[track_number],
			"duration_ms":[duration_ms],
			"key":[key],
			"mode":[mode],
			"time_signature":[time_signature],
			"acousticness":[acousticness],
			"danceability":[danceability],
			"energy":[energy],
			"instrumentalness":[instrumentalness],
			"loudness":[loudness],
			"speechiness":[speechiness],
			"valence":[valence],
			"tempo":[tempo],
			"explicit":[explicit],
			"complexity":[complexity],
			"popularity":[popularity]
			})

		df = df.append(song_data, ignore_index = True)

		return df