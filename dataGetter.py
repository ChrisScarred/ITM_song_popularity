import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

# obtained from Spotify Developer API
SPOTIPY_CLIENT_ID='17fc2297bbdf4be3b339cf17f8307c45'
SPOTIPY_CLIENT_SECRET='46eaf0adb7c8472994ed2404c7700142'
SPOTIPY_REDIRECT_URI='http://localhost:8888/callback'

'''
Obtains song info from:
artist identified by artist_uri
exluding albums non_inc_albs
excluding tracks non_inc_tracks
and saves into file with name *name*.csv
'''
class DataGetter:
	def __init__(self, artist_uri, non_inc_albs, non_inc_tracks, name):
		# initialisation
		self.artist_uri = artist_uri
		self.non_inc_tracks = non_inc_tracks
		self.non_inc_albs = non_inc_albs
		self.name = name
		# connects to spotify
		self.spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, 
			client_secret=SPOTIPY_CLIENT_SECRET))		

	# gets the song data
	def getData(self):
		# gets all albums
		album_results = self.spotify.artist_albums(self.artist_uri, album_type='album')
		albums = album_results['items']

		# prepares song df
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
	
		# loops through album iterator, gets albums items (songs) of every album object
		while album_results['next']:
			album_results = self.spotify.next(album_results)
			albums.extend(album_results['items'])

		# loops thorugh the list of album items (songs)
		for album in albums:
			# checks if album should be included
			album_name = album['name']
			if all([album_name != x for x in self.non_inc_albs]): 
				album_id = album['id']

				print("Processing album "+album_name)			

				# obtains the song info
				track_results = self.spotify.album_tracks(album_id)
				# obtains the items of songs info
				tracks = track_results['items']

				# saves the items to array
				while track_results['next']:
					track_results = self.spotify.next(track_results)
					tracks.extend(track_results['items'])

				# processes all songs
				for track in tracks:
					track_id = track['id']
					track_name = track['name']
					# checks if should be included
					if all(track_name != x for x in self.non_inc_tracks):
						print('- - Processing track '+track_name)
						# appends new song data to songs_data df
						songs_data = self.getSongData(songs_data, track_name, track_id)
		# saves the data to csv
		songs_data.to_csv(self.name+".csv")			
		print("Done. Results saved to file "+self.name+".csv")

		return songs_data

	def getSongData(self, df, name, track_id):
		# obtains wanted results
		features = self.spotify.audio_features(tracks=[track_id])[0]
		track_data = self.spotify.track(track_id)
		analysis = self.spotify.audio_analysis(track_id)

		# prepares variables
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

		# creates df
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

		# appends new df to all data df 
		df = df.append(song_data, ignore_index = True)

		return df