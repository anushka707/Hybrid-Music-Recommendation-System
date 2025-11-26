from src.hybrid_engine import HybridRecommender

hybrid = HybridRecommender(alpha=0.7)
print(hybrid.recommend(user_id=2))

