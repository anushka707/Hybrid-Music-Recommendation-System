import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from .config import DATA_RAW, EMBEDDING_DIR

class ArtistEmbeddingGenerator:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def load_data(self):
        artists = pd.read_csv(f"{DATA_RAW}/artists.dat", sep="\t", encoding="latin1")
        tags = pd.read_csv(f"{DATA_RAW}/tags.dat", sep="\t", encoding="latin1")
        user_tags = pd.read_csv(f"{DATA_RAW}/user_taggedartists.dat", sep="\t", encoding="latin1")

        return artists, tags, user_tags

    def prepare_text_corpus(self):
        artists, tags, user_tags = self.load_data()

        # artists.dat has "id" instead of "artistID"
        artists = artists.rename(columns={"id": "artistID"})

        # Build tag text per artist
        tag_map = (
            user_tags.merge(tags, on="tagID")
                     .groupby("artistID")["tagValue"]
                     .apply(lambda x: " ".join(x))
                     .reset_index()
        )

        df = artists.merge(tag_map, on="artistID", how="left")
        df["text"] = df["name"] + " " + df["tagValue"].fillna("")

        return df[["artistID", "text"]]

    def generate_embeddings(self):
        df = self.prepare_text_corpus()

        embeddings = self.model.encode(df["text"].tolist(), show_progress_bar=True)

        os.makedirs(EMBEDDING_DIR, exist_ok=True)
        np.save(os.path.join(EMBEDDING_DIR, "artist_embeddings.npy"), embeddings)
        np.save(os.path.join(EMBEDDING_DIR, "artist_ids.npy"), df["artistID"].values)

        print("\n🎵 Artist embeddings saved successfully!\n")

        return embeddings
