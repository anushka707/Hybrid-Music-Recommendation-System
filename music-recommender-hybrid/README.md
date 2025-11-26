# 🎵 Hybrid Music Recommendation System  
### _A PySpark + Sentence-BERT Based Personalized Music Recommender_

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PySpark](https://img.shields.io/badge/PySpark-3.x-orange.svg)
![NLP](https://img.shields.io/badge/Sentence--BERT-Embeddings-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A hybrid music recommendation system built using **PySpark (ALS)** for collaborative filtering  
and **Sentence-BERT embeddings** for content-based similarity.  
The system intelligently combines both signals to produce **highly relevant song recommendations**  
based on user behaviour, artist preferences, and track metadata (title, artist, tags).

This project is designed for **academic submission**,  
**portfolio showcasing**, and **real-world recommender system learning**.

---

# 🚀 Features

### ✔ Collaborative Filtering (PySpark ALS)
- Learns latent patterns from user–artist interactions  
- Generates top-N personalized recommendations  
- Optimized with RMSE-based evaluation  

### ✔ Content-Based Filtering (Sentence-BERT)
- Generates semantic embeddings for track descriptions  
- Captures similarity in tags, artists, and titles  
- Enables similarity-based recommendations  

### ✔ Hybrid Engine
- Weighted combination:  
  **`Hybrid Score = α * ALS + β * Cosine Similarity`**
- Produces richer, more diverse recommendations  
- Mimics real-world systems like Spotify + Last.fm  

### ✔ Evaluation Metrics
- RMSE  
- Precision@K  
- Recall@K  
- MAP@K  
- NDCG@K  

### ✔ Four Jupyter Notebooks  
For clean walkthrough and demonstrations.

---

# 📂 Project Structure

music-recommender-hybrid/
│
├── data/
│   ├── raw/
│   │   ├── artists.dat
│   │   ├── tags.dat
│   │   ├── user_artists.dat
│   │   ├── user_taggedartists.dat
│   │   ├── user_friends.dat
│
├── models/
│   ├── als/
│   ├── embeddings/
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── als_model.py
│   ├── embeddings.py
│   ├── hybrid_recommender.py
│
├── test_als.py
├── test_embeddings.py
├── test_hybrid.py



---

# 📦 Dataset (HetRec 2011 - Last.fm)
Dataset link: https://grouplens.org/datasets/hetrec-2011/

Files included:
- `artists.dat`  
- `tracks.dat`  
- `tags.dat`  
- `user_artists.dat`  
- `user_taggedtracks.dat`  
- `user_taggedartists.dat`  

---

# 🔧 Installation

### 1. Clone the repo:
**`git clone https://github.com/anushka707/Hybrid-Music-Recommendation-System`**
**`cd Hybrid-Music-Recommendation-System`**

### 2. Create virtual environment:
**`python3 -m venv .venv`**
**`source .venv/bin/activate`**

### 3. Install dependencies:
**`pip install -r requirements.txt`**

### 4. Place dataset in:
**`data/raw/`**


# ⚙️ Running the System

### ALS:
**`python3 test_als.py`**

### Embeddings:
**`python3 test_embeddings.py`**

### Hybrid:
**`python3 test_hybrid.py`**


# 🧠 How the Model Works
### ALS (Collaborative Filtering)
Learns user–artist interaction patterns using PySpark’s Alternating Least Squares.

### Sentence-BERT Embeddings
Generates semantic vector representations of:
**`track title + artist name + tags`**

### Hybrid Recommendation
**`final_score = α * als_score + β * cosine_similarity`**

# 📊 Evaluation Metrics

### Implemented metrics:
**`RMSE`**
**`Precision@K`**
**`Recall@K`**
**`MAP@K`**
**`NDCG@K`**