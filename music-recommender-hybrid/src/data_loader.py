import pandas as pd
from .config import DATA_RAW

class LastFMDataset:
    def load_user_artists(self):
        return pd.read_csv(f"{DATA_RAW}/user_artists.dat", sep="\t", header=0)

    def load_artists(self):
        return pd.read_csv(f"{DATA_RAW}/artists.dat", sep="\t", header=0)

    def load_tags(self):
        return pd.read_csv(f"{DATA_RAW}/tags.dat", sep="\t", header=0)

    def load_user_tags(self):
        return pd.read_csv(f"{DATA_RAW}/user_taggedartists.dat", sep="\t", header=0)
