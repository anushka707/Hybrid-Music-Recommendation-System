import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.evaluation import RegressionEvaluator

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.config import DATA_RAW, EMBEDDING_DIR, ALS_MODEL_DIR
from src.hybrid_engine import HybridRecommender


# ---------------------------------------
# SETUP
# ---------------------------------------
os.makedirs("results", exist_ok=True)

spark = (
    SparkSession.builder.master("local[*]")
    .appName("ResultsGenerator")
    .getOrCreate()
)

# Load ALS model
als_model = ALSModel.load(ALS_MODEL_DIR)


# ---------------------------------------
# 1. DISTRIBUTION OF LISTENING COUNTS
# ---------------------------------------
print("Plotting play count distribution...")

df = pd.read_csv(f"{DATA_RAW}/user_artists.dat", sep="\t")

plt.figure(figsize=(8, 5))
sns.histplot(df["weight"], bins=50, kde=True, color="blue")
plt.title("Distribution of Listening Counts")
plt.xlabel("Play Count (weight)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("results/playcount_distribution.png")
plt.close()


# ---------------------------------------
# 2. ALS RMSE CALCULATION
# ---------------------------------------
print("Calculating ALS RMSE...")

als_df = spark.read.csv(
    f"{DATA_RAW}/user_artists.dat",
    sep="\t", header=True, inferSchema=True
)

train, test = als_df.randomSplit([0.8, 0.2], seed=42)

preds = als_model.transform(test)

evaluator = RegressionEvaluator(
    metricName="rmse", labelCol="weight", predictionCol="prediction"
)

rmse_value = evaluator.evaluate(preds)

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["RMSE"], [rmse_value], color="green")
ax.set_title("ALS RMSE on Test Set")
plt.tight_layout()
plt.savefig("results/als_rmse.png")
plt.close()


# ---------------------------------------
# 3. PCA OF EMBEDDINGS
# ---------------------------------------
print("Generating PCA of embeddings...")

embeddings = np.load(f"{EMBEDDING_DIR}/artist_embeddings.npy")

# Reduce to 2D
pca = PCA(n_components=2)
reduced_pca = pca.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(reduced_pca[:, 0], reduced_pca[:, 1], s=2, alpha=0.6)
plt.title("Artist Embeddings (PCA 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("results/embeddings_pca.png")
plt.close()


# ---------------------------------------
# 4. Hybrid Recommendations Heatmap
# ---------------------------------------
print("Generating hybrid recommendation heatmap...")

hybrid = HybridRecommender(alpha=0.7)
results = hybrid.recommend(user_id=2, top_k=10)

artist_ids = [x[0] for x in results]
scores = [float(x[1]) for x in results]

plt.figure(figsize=(10, 4))
sns.heatmap(
    np.array(scores).reshape(1, -1),
    annot=True, fmt=".1f", cmap="viridis"
)
plt.title("Hybrid Recommendation Scores (User 2)")
plt.yticks([])
plt.xlabel("Artist Rank")
plt.tight_layout()
plt.savefig("results/hybrid_scores.png")
plt.close()


# ---------------------------------------
# 5. CF Score vs Content Similarity
# ---------------------------------------
print("Generating CF vs Content similarity correlation plot...")

cf_scores_dict = dict(results)

# content similarities from embeddings
seed_embedding = embeddings[np.where(np.load(f"{EMBEDDING_DIR}/artist_ids.npy") == artist_ids[0])[0][0]]

similarities = []
for aid in artist_ids:
    idx = np.where(np.load(f"{EMBEDDING_DIR}/artist_ids.npy") == aid)[0][0]
    sim = np.dot(seed_embedding, embeddings[idx]) / (
        np.linalg.norm(seed_embedding) * np.linalg.norm(embeddings[idx])
    )
    similarities.append(sim)

plt.figure(figsize=(6, 6))
plt.scatter(list(cf_scores_dict.values()), similarities, color="purple")
plt.xlabel("CF Score")
plt.ylabel("Content Similarity")
plt.title("Correlation: ALS Score vs Embedding Similarity")
plt.tight_layout()
plt.savefig("results/cf_vs_content_scatter.png")
plt.close()

# ---------------------------------------
# 6. t-SNE Visualization of Embeddings
# ---------------------------------------
print("Generating t-SNE visualization (may take 1–2 minutes)...")
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate='auto',
    max_iter=1500,
    init="pca"
)

emb_tsne = tsne.fit_transform(embeddings[:5000])  # run on 5K points for speed

plt.figure(figsize=(8, 6))
plt.scatter(emb_tsne[:, 0], emb_tsne[:, 1], s=3, cmap="Spectral")
plt.title("Artist Embeddings (t-SNE, 5k sample)")
plt.tight_layout()
plt.savefig("results/embeddings_tsne.png")
plt.close()


# ---------------------------------------
# 7. UMAP Visualization
# ---------------------------------------
print("Generating UMAP visualization...")

import umap

umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine")
emb_umap = umap_model.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(emb_umap[:, 0], emb_umap[:, 1], s=3, alpha=0.7)
plt.title("Artist Embeddings (UMAP 2D)")
plt.tight_layout()
plt.savefig("results/embeddings_umap.png")
plt.close()


# ---------------------------------------
# 8. User-to-User Similarity Graph
# ---------------------------------------
print("Generating user-to-user similarity graph...")

# Create a pivot: rows = users, columns = artists
pivot = df.pivot_table(index="userID", columns="artistID", values="weight", fill_value=0)

# Compute cosine similarities between users
from sklearn.metrics.pairwise import cosine_similarity
user_sim = cosine_similarity(pivot)

# Build a similarity graph (only edges > 0.5)
import networkx as nx

G = nx.Graph()

user_ids = pivot.index.tolist()

for i in range(len(user_ids)):
    for j in range(i + 1, len(user_ids)):
        if user_sim[i, j] > 0.5:
            G.add_edge(user_ids[i], user_ids[j], weight=user_sim[i, j])

# Plot the graph
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx_nodes(G, pos, node_size=15, node_color="blue")
nx.draw_networkx_edges(G, pos, alpha=0.15)
plt.title("User–User Similarity Network (edges > 0.5)")
plt.axis("off")
plt.savefig("results/user_similarity_graph.png")
plt.close()



# ---------------------------------------
# 9. Artist Clusters using K-Means on Embeddings
# ---------------------------------------
print("Generating artist clusters...")

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10, random_state=42)
labels = kmeans.fit_predict(embeddings)

pca_cluster = PCA(n_components=2).fit_transform(embeddings)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=pca_cluster[:, 0],
    y=pca_cluster[:, 1],
    hue=labels,
    palette="tab10",
    s=5,
    alpha=0.7,
    legend=False
)
plt.title("Artist Embedding Clusters (K-Means + PCA Projection)")
plt.tight_layout()
plt.savefig("results/artist_clusters.png")
plt.close()



print("\n🎉 ALL RESULTS GENERATED SUCCESSFULLY!")
print("Check the results/ folder.")
