import sys

sys.path.append("..")

import torch
import torch.nn as N
import torch.nn.functional as F
from data_split_and_history import prep_hist_n_data
from src.utils.config import EMBS_ITEM, EMBS_USER

prep_feature_embs = torch.load(EMBS_ITEM, weights_only=True)
prep_user_embs = torch.load(EMBS_USER,weights_only=True)



class ItemTower(N.Module):
    def __init__(
        self,
        num_of_rows: int,
        lay_sz: int,
        prep_tens_features: torch.tensor,
        num_of_features: int,
        output_sz: int,
        id_lay_sz: int,
    ):
        super().__init__()
        self.id_embs = N.Embedding(num_of_rows, id_lay_sz)
        self.input_size = id_lay_sz + num_of_features

        self.ready_feature_embs = N.Embedding.from_pretrained(
            prep_tens_features, freeze=True
        )

        self.head = N.Sequential(
            N.Linear(self.input_size, lay_sz),
            N.BatchNorm1d(lay_sz),
            N.ReLU(),
            N.Dropout(0.15),
            N.Linear(lay_sz, output_sz),
        )

    def forward(self, m_id: torch.Tensor) -> torch.Tensor:
        v_id = self.id_embs(m_id)
        if self.training:
            binary_val = (torch.rand(m_id.shape[0], device=m_id.device) > 0.2).float()
            v_id = v_id * binary_val.unsqueeze(1)

        v_ready = self.ready_feature_embs(m_id)
        out = self.head(torch.cat((v_ready, v_id), dim=-1))
        return F.normalize(out, p=2, dim=-1)

class UserTower(N.Module):
    def __init__(self, prep_user: torch.Tensor, num_users: int, id_lay_sz: int, lay_sz: int, output_sz: int, hist_dim: int, proj_hist_sz: int):
        super().__init__()
        self.ids_embs = N.Embedding(num_users, id_lay_sz)
        self.prep_us_emb=N.Embedding.from_pretrained(prep_user, freeze= True)
        self.history_proj=N.Sequential(
                    N.Linear(hist_dim,proj_hist_sz),
                    N.BatchNorm1d(proj_hist_sz),
                    N.ReLU(),
                    N.Dropout(0.15)
                )
        self.input_size=id_lay_sz+prep_user.shape[1]+proj_hist_sz
        self.head=N.Sequential(
            N.Linear(self.input_size, lay_sz),
            N.BatchNorm1d(lay_sz),
            N.ReLU(),
            N.Dropout(0.15),
            N.Linear(lay_sz, output_sz)
        )
    def forward(self, history_embs: torch.Tensor, id_user: torch.Tensor) -> torch.Tensor:
        v_id=self.ids_embs(id_user)
        v_hist=self.history_proj(history_embs)
        v_prep=self.prep_us_emb(id_user)

        concat_features=torch.cat((v_id,v_hist,v_prep),dim=-1)

        out=self.head(concat_features)

        return F.normalize(out,p=2,dim=-1)