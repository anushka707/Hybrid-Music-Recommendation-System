from src.evaluation import precision_at_k, recall_at_k, mapk, ndcg_at_k

recommended = [10, 20, 30, 40, 50]
actual = [30, 50, 70]

print("Precision@5:", precision_at_k(recommended, actual, k=5))
print("Recall@5:", recall_at_k(recommended, actual, k=5))
print("MAP@5:", mapk([actual], [recommended], k=5))
print("NDCG@5:", ndcg_at_k(recommended, actual, k=5))
