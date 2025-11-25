import pandas as pd
from .config import DATA_RAW

def load_user_artists():
    return pd.read_csv(f"{DATA_RAW}/user_artists.dat", sep='\t')

def load_artists():
    return pd.read_csv(f"{DATA_RAW}/artists.dat", sep='\t')

def load_tracks():
    return pd.read_csv(f"{DATA_RAW}/tracks.dat", sep='\t')

def load_user_tagged_tracks():
    return pd.read_csv(f"{DATA_RAW}/user_taggedtracks.dat", sep='\t')

def load_tags():
    return pd.read_csv(f"{DATA_RAW}/tags.dat", sep='\t')
