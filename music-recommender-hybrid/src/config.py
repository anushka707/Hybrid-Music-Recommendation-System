import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR = os.path.join(BASE_DIR, "models")

ALS_MODEL_DIR = os.path.join(MODEL_DIR, "als")
EMBEDDING_DIR = os.path.join(MODEL_DIR, "embeddings")

os.makedirs(ALS_MODEL_DIR, exist_ok=True)
os.makedirs(EMBEDDING_DIR, exist_ok=True)
