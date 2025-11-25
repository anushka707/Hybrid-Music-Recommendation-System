import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')

ALS_MODEL_DIR = os.path.join(BASE_DIR, 'models', 'als')
EMBEDDINGS_DIR = os.path.join(BASE_DIR, 'models', 'embeddings')
