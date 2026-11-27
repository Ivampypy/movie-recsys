import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.extend([str(ROOT_DIR), str(CURRENT_DIR)])


import faiss
from TTR_model import TwoTowerRetrieval
from src.utils.config import EMBS_ITEM, EMBS, EMBS_USER
import torch

prep_feature_embs = torch.load(EMBS_ITEM, weights_only=True)
prep_user_embs = torch.load(EMBS_USER, weights_only=True)


device = "cpu"
temperature = 0.07
num_us = 943
num_items = 1682
hist_dim = 384
lay_sz = 64
id_lay_sz = 16
output_sz = 64
proj_hist_sz = 64
features = prep_feature_embs.shape[1]

model = TwoTowerRetrieval(
    temperature,
    prep_user_embs,
    prep_feature_embs,
    num_us,
    num_items,
    id_lay_sz,
    lay_sz,
    output_sz,
    hist_dim,
    proj_hist_sz,
    features,
).to(device)

model.load_state_dict(torch.load(EMBS / "TTR_weights.pt", weights_only=True))

model.eval()

films_vec = model.item_tower(torch.arange(num_items))
films_vec = films_vec.detach().cpu().numpy().astype("float32")

index = faiss.IndexFlatIP(64)
index.add(films_vec)


faiss.write_index(index, "models/movies.index")
