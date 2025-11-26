from src.als_model import ALSRecommender

recommender = ALSRecommender()
recommender.train()
print(recommender.recommend_for_user(2))
