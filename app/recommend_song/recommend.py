import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pickle
import pandas as pd
import numpy as np
import random
from ..clean_text.clean_text import clean_doc

cid = 'b584f71041ba4c43a58699377b25d141'
secret = '85c37c96814a4cacbf90021f1045e087'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

playlists = {
             'sad' : ["https://open.spotify.com/playlist/6eq5Zviu57AxTT02x0F3pE?si=3b8fb3bf44b8453c"],
             'joy':  ["https://open.spotify.com/playlist/4nd7oGDNgfM0rv28CQw9WQ?si=tY_dRo5RSXKPfbbckybheQ&utm_source=whatsapp&nd=1"],
             'love': ["https://open.spotify.com/playlist/1BOW7T7H25BQnu3T26JEXP?si=lmN5k8NuTQu6wNU8PHT7TA&utm_source=whatsapp&nd=1"],
            'angry':  ["https://open.spotify.com/playlist/5cwtgqs4L1fX8IKoQebfjJ?si=MCSvIfm7RXW8dNOmhTWABw&utm_source=whatsapp&nd=1"]

}
hindi_df = pd.read_csv('./app/recommend_song/tracks2.csv')
hindi_df_r = hindi_df.drop(['Unnamed: 0','id'],axis=1)


with open('./app/recommend_song/distance.pkl','rb') as file1:
  distances = pickle.load(file1)
with open('./app/pickle_files/indices.pkl','rb') as file2:
  indices = pickle.load(file2)
with open('./app/pickle_files/model_lr','rb') as file3:
  model = pickle.load(file3)
with open('./app/pickle_files/model_lr_cv','rb') as file4:
  lr_cv = pickle.load(file4)




def get_link(id):
  # spotify.track(track_id=id,market="AR")['name']
  return {id:[spotify.track(track_id=id,market="AR")['name'],spotify.track(track_id=id,market="AR")['external_urls']['spotify'],spotify.track(track_id=id,market="AR")['album']['images'][0]['url']]}

def song_of_mood(playlist,mood):
        link = playlist[mood]
        id= link[0][34:56]
      
        
        pl_tracks = spotify.playlist_tracks(id)['items']
        ids = [foo['track']['id'] for foo in pl_tracks]
        sid = random.choices(ids,k=6)
        song_lst = [get_link(id) for id in sid if spotify.track(track_id=id,market="AR")['name'] != ""]

        for id in sid:
           song_lst.extend(spot_next(id))
        # song_lst_ = [next(list(s.keys())[0]) for s  in song_lst ]
        
        return song_lst
 

def get_mood(text):
    
  pred = model.predict(lr_cv.transform(clean_doc([text])))
  dict_ = {0: 'sad', 1: 'joy', 2: 'love', 3: 'angry', 4: 'fear', 5: 'surprise'}
  mood = dict_[pred[0]]
  print(f"mood detected : {mood}")
  links = song_of_mood(playlists,mood)
  return links,mood


def next(id):
    idx = hindi_df.index[hindi_df['id']==id]
    lst = indices[idx][0]
    lst_id = [hindi_df.loc[lst[i]]['id'] for i in range(1,3)]
    song_lst = [get_link(id) for id in lst_id if spotify.track(track_id=id,market="AR")['name'] != ""]
    return song_lst

def spot_next(id):
  lst = [spotify.recommendations(seed_tracks=[id],limit=2)['tracks'][i]['id'] for i in range(2)]
  song_lst = [get_link(id) for id in lst if spotify.track(track_id=id,market="AR")['name'] != ""]
  return song_lst