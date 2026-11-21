import torch
from collections import defaultdict
from math import log2

@torch.no_grad()
def evaluation_retrieval(model, k: int, train_data: torch.Tensor, test_data: torch.Tensor, num_items: int, device: str):
    model.eval()

    all_it_vecs=model.item_tower(torch.arange(num_items,device=device))

    seen_movies=defaultdict(set)

    hits,ndcg_sum=0.0,0.0

    for user_ids, movies, _ in train_data:
        seen_movies[user_ids].add(movies)

    for user_ids, movies, hist_emb in test_data:
        u_vec=model.user_tower(hist_emb.unsqueeze(0).to(device=device),
                               torch.tensor([user_ids],dtype=torch.long,device=device))
        scores = (u_vec@all_it_vecs.T).squeeze(0)

        for seen_m in seen_movies[user_ids]:
            if seen_m!=movies:
                scores[seen_m] = -1e8

        top_k_items=torch.topk(scores,k=k).indices.tolist()

        if movies in top_k_items:
            hits+=1.0
            rank=top_k_items.index(movies)
            ndcg_sum+=1.0/log2(rank+2)


    recall_at_k=hits/len(test_data)
    ndcg_at_k=ndcg_sum/len(test_data)

    return recall_at_k, ndcg_at_k




