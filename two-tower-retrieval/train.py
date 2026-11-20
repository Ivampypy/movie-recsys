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
dl =DataLoader(dset, batch_size=64,shuffle=True)

