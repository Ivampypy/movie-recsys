from TTR_model import TwoTowerRetrieval
import torch
import torch.nn as N
import torch.nn.functional as F
from src.utils.config import EMBS_ITEM, EMBS_USER
from data_split_and_history import MovieRetrievalDataset, prep_hist_n_data
from torch.utils.data import DataLoader

prep_feature_embs = torch.load(EMBS_ITEM, weights_only=True)
prep_user_embs = torch.load(EMBS_USER,weights_only=True)
train_data, test_data =prep_hist_n_data(prep_feature_embs)

dset=MovieRetrievalDataset(train_data)
dl=DataLoader(dset, batch_size=64,shuffle=True)

device='cuda'
num_epochs = 1000
lr=1e-3
num_us=943
num_items=1682
hist_dim=64
lay_sz=128
id_lay_sz=64
output_sz=64
proj_hist_sz=32
features=prep_feature_embs.shape[1]

model=TwoTowerRetrieval(prep_user_embs,prep_feature_embs,num_us,num_items,id_lay_sz,lay_sz,output_sz,hist_dim,proj_hist_sz,features).to(device)

optimizer=torch.optim.AdamW(model.parameters(), lr=lr)


