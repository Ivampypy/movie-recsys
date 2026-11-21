import torch
from collections import defaultdict
from math import log2

@torch.no_grad()
def evaluation_retrieval(model, k: int, train_data: torch.Tensor, test_data: torch.Tensor, num_items: int, device: str):
    model.eval()

    all_it_vecs=model.item_tower(torch.arange(num_items,device=device))

    seen_movies=defaultdict(set)
    rel_movies=defaultdict(set)

    last_hist=dict()

    rec,ndcg=list(),list()

    for user_id, m_id, hist_emb in train_data:
        seen_movies[user_id].add(m_id)
        last_hist[user_id] = hist_emb
    
    for user_id, rel_id, hist_emb in test_data:
        if user_id in last_hist.keys():
            rel_movies[user_id].add(rel_id)

    
    for key, value in rel_movies.items():
        if key in last_hist.keys():
            u_vec=model.user_tower(last_hist[key].unsqueeze(0).to(device),torch.tensor([key],dtype=torch.long,device=device) )
            scores=(u_vec@all_it_vecs.T).squeeze(0)
            for seen_m in seen_movies[key]:
                scores[seen_m] = -1e8
            top_k_movies=torch.topk(scores,k=k).indices.tolist()

            dcg=sum(1.0/log2(pos+1+1) for pos,ind in enumerate(top_k_movies) if ind in value)
            idcg=sum(1.0/log2(pos+1+1) for pos in range(min(k,len(value))))
                
            rec.append(len(set(top_k_movies) & value)/len(value))
            ndcg.append(dcg/idcg if idcg>0 else 0.0)
    recall_at_k=sum(rec)/len(rec)
    ndcg_at_k=sum(ndcg)/len(ndcg)

    return recall_at_k, ndcg_at_k            







