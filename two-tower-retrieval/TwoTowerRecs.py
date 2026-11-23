import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.extend([str(ROOT_DIR), str(CURRENT_DIR)])

from TTR_model import TwoTowerRetrieval
from src.utils.config import EMBS_ITEM, EMBS_USER, INDEX_FILE, TTR_CONFIG
import torch
import faiss
from collections import defaultdict
from src.data.load import load_ratings
import pandas as pd


class TwoTowerRecommender:
    def __init__(
        self,
        state_dict_path: Path,
        index_path: Path,
        embs_item_path: Path,
        embs_user_path: Path,
    ):
        self.prep_it_embs = torch.load(embs_item_path, weights_only=True)
        self.prep_us_embs = torch.load(embs_user_path, weights_only=True)
        self.model = TwoTowerRetrieval(
            TTR_CONFIG["temperature"],
            self.prep_us_embs,
            self.prep_it_embs,
            TTR_CONFIG["num_users"],
            TTR_CONFIG["num_items"],
            TTR_CONFIG["id_lay_sz"],
            TTR_CONFIG["lay_sz"],
            TTR_CONFIG["output_sz"],
            TTR_CONFIG["hist_dim"],
            TTR_CONFIG["proj_hist_sz"],
            self.prep_it_embs.shape[1],
        )

        self.index = faiss.read_index(str(index_path))
        self.model.load_state_dict(torch.load(state_dict_path, weights_only=True))

        self.model.eval()
        self.user_seen = defaultdict(set)
        self.user_history = defaultdict(list)

    def load(self, ratings: pd.DataFrame):
        ratings = ratings.sort_values(by="timestamp")
        for ind, row in ratings.iterrows():
            rate = row["rating"]
            us_id = row["user_id"]
            m_id = row["movie_id"]
            if rate >= 4:
                self.user_history[us_id].append(m_id)
            self.user_seen[us_id].add(m_id)

    def recommend(self, us_id: int, top_n: int = 10) -> list[int]:
        if us_id in self.user_history:
            past_movies=self.user_history[us_id][-5:]
            past_movies_idx=[m-1 for m in past_movies]
            hist_emb=self.prep_it_embs[past_movies_idx,1:].mean(dim=0,keepdim=True)
            u_vec=self.model.user_tower(hist_emb,torch.tensor([us_id-1],dtype=torch.long))
            u_vec=u_vec.detach().cpu().numpy().astype('float32')
            _,inds=self.index.search(u_vec,k=top_n+len(self.user_seen[us_id]))
            filtered=[m+1 for m in inds[0] if m+1 not in self.user_seen[us_id]]
            return filtered[:top_n]



        return self.recommend_cold_start()

    def recommend_cold_start(self):
        return None

