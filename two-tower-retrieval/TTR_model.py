import sys

sys.path.append("..")

import torch
import torch.nn as N
import torch.nn.functional as F

from src.utils.config import EMBS_ITEM, EMBS_USER

prep_feature_embs = torch.load(EMBS_ITEM, weights_only=True)


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
            N.ReLu(),
            N.Dropout(0.15),
            N.Linear(lay_sz, output_sz),
        )

    def forward(self, m_id: torch.Tensor) -> torch.Tensor:
        v_id = self.id_embs(m_id)
        if self.training:
            binary_val = (torch.rand(m_id.shape[0], device=m_id.device) > 0.2).float()
            v_id = v_id * binary_val

        v_ready = self.ready_feature_embs(m_id)
        out = self.head(torch.cat((v_ready, v_id), dim=-1))
        return F.normalize(out, p=2, dim=-1)
