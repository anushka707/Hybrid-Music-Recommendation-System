import os

# -------- CHANGE THIS TO WHATEVER NAME YOU WANT --------
PROJECT_NAME = "music-recommender-hybrid"
# --------------------------------------------------------

folders = [
    f"{PROJECT_NAME}/data/raw",
    f"{PROJECT_NAME}/data/processed",
    f"{PROJECT_NAME}/notebooks",
    f"{PROJECT_NAME}/src",
    f"{PROJECT_NAME}/models/als",
    f"{PROJECT_NAME}/models/embeddings",
    f"{PROJECT_NAME}/results/metrics",
    f"{PROJECT_NAME}/results/figures",
]

files = {
    f"{PROJECT_NAME}/README.md": "# Hybrid Music Recommendation System (PySpark ALS + NLP)\n",
    f"{PROJECT_NAME}/requirements.txt": """pyspark
pandas
numpy
matplotlib
seaborn
scikit-learn
sentence-transformers
tqdm
plotly
""",
    f"{PROJECT_NAME}/.gitignore": """data/raw/*
data/processed/*
models/*
results/*
*.pyc
__pycache__/
.ipynb_checkpoints/
venv/
.env
""",
    f"{PROJECT_NAME}/LICENSE": "MIT License\n\n[Add your name and year here]",
    f"{PROJECT_NAME}/src/__init__.py": "",
    f"{PROJECT_NAME}/src/config.py": """import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')

ALS_MODEL_DIR = os.path.join(BASE_DIR, 'models', 'als')
EMBEDDINGS_DIR = os.path.join(BASE_DIR, 'models', 'embeddings')
""",
    f"{PROJECT_NAME}/src/data_loader.py": """import pandas as pd
from .config import DATA_RAW

def load_user_artists():
    return pd.read_csv(f"{DATA_RAW}/user_artists.dat", sep='\\t')

def load_artists():
    return pd.read_csv(f"{DATA_RAW}/artists.dat", sep='\\t')

def load_tracks():
    return pd.read_csv(f"{DATA_RAW}/tracks.dat", sep='\\t')

def load_user_tagged_tracks():
    return pd.read_csv(f"{DATA_RAW}/user_taggedtracks.dat", sep='\\t')

def load_tags():
    return pd.read_csv(f"{DATA_RAW}/tags.dat", sep='\\t')
""",
}

def create_structure():
    # Create folders
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")

    # Create files
    for filepath, content in files.items():
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created file: {filepath}")

    print("\n🎉 Project structure successfully created!")
    print(f"📁 Folder: {PROJECT_NAME}")

if __name__ == "__main__":
    create_structure()
