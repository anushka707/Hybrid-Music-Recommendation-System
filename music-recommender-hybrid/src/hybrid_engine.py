import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from .config import ALS_MODEL_DIR, EMBEDDING_DIR

class HybridRecommender:
    def __init__(self, alpha=0.7):
        self.alpha = alpha

        # Initialize Spark session (required before loading ALS model)
        self.spark = (
            SparkSession.builder
            .appName("HybridRecommender")
            .master("local[*]")
            .getOrCreate()
        )

        # Load ALS model
        self.cf_model = ALSModel.load(ALS_MODEL_DIR)

        # Load embeddings
        self.embeddings = np.load(f"{EMBEDDING_DIR}/artist_embeddings.npy")
        self.artist_ids = np.load(f"{EMBEDDING_DIR}/artist_ids.npy")

    def recommend(self, user_id, top_k=10):
        # Prepare user dataframe
        user_df = self.spark.createDataFrame([(user_id,)], ["userID"])

        # CF recommendations
        cf_df = self.cf_model.recommendForUserSubset(user_df, top_k).collect()

        if not cf_df:
            return []

        cf_items = cf_df[0]["recommendations"]

        # Convert CF items to dict
        cf_scores = {item["artistID"]: item["rating"] for item in cf_items}

        # Pick the highest CF-score artist as the "seed" for similarity
        seed_artist = list(cf_scores.keys())[0]
        seed_idx = np.where(self.artist_ids == seed_artist)[0][0]

        # Compute cosine similarity
        sim_scores = cosine_similarity(
            self.embeddings[seed_idx].reshape(1, -1),
            self.embeddings
        )[0]

        # Hybrid score = α * CF + (1-α) * content similarity
        hybrid_scores = {}
        for artist_id, cf_score in cf_scores.items():
            idx = np.where(self.artist_ids == artist_id)[0][0]
            hybrid_scores[artist_id] = (
                self.alpha * cf_score + (1 - self.alpha) * sim_scores[idx]
            )

        # Sort and return
        ranked = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
