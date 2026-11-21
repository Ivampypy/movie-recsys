import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.extend([str(ROOT_DIR), str(CURRENT_DIR)])
from TTR_model import TwoTowerRetrieval
import torch
import torch.nn as N
import torch.nn.functional as F
from src.utils.config import EMBS_ITEM, EMBS_USER
from data_split_and_history import MovieRetrievalDataset, prep_hist_n_data
from torch.utils.data import DataLoader
from src.evaluation.eval_retrieval import evaluation_retrieval

prep_feature_embs = torch.load(EMBS_ITEM, weights_only=True)
prep_user_embs = torch.load(EMBS_USER,weights_only=True)
train_data, test_data =prep_hist_n_data(prep_feature_embs)

dsettr=MovieRetrievalDataset(train_data)
dsetvl=MovieRetrievalDataset(test_data)

dl_tr=DataLoader(dsettr, batch_size=256//2,shuffle=True)
dl_test=DataLoader(dsetvl,batch_size=256//2,shuffle=True)

device='cpu'
num_epochs = 100
lr=1e-3
temperature=0.07
num_us=943
num_items=1682
hist_dim=384
lay_sz=64
id_lay_sz=16
output_sz=64
proj_hist_sz=64
features=prep_feature_embs.shape[1]

model=TwoTowerRetrieval(temperature,prep_user_embs,prep_feature_embs,num_us,num_items,id_lay_sz,lay_sz,output_sz,hist_dim,proj_hist_sz,features).to(device)

optimizer=torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
if '__name__'=='__main__':
    best_rec=0.0; ndcg = 0.0
    for epoch in range(num_epochs):
        model.train()
        total_loss=0.0
        for _,tuple_ds in enumerate(dl_tr):
            user_ids=tuple_ds[0].to(device)
            movies=tuple_ds[1].to(device)
            history=tuple_ds[2].to(device)
            optimizer.zero_grad()

            logits=model(user_ids,movies,history)
            labels = torch.arange(user_ids.shape[0]).to(device)
            loss=F.cross_entropy(logits,labels)

            loss.backward()
            optimizer.step()

            total_loss+=loss.item()

        scheduler.step()

        if epoch%5==0:

            rec10,ndcg10=evaluation_retrieval(
                model,
                10,
                train_data,
                test_data,
                num_items,
                'cpu'
            )
            """ for _,tuple_ds in enumerate(dl_test):
                    user_ids=tuple_ds[0].to(device)
                    movies=tuple_ds[1].to(device)
                    history=tuple_ds[2].to(device)
                    logits=model(user_ids,movies,history)
                    labels = torch.arange(user_ids.shape[0]).to(device)
                    test_loss=F.cross_entropy(logits,labels)
                    test_tloss += test_loss.item()
                    """
            if rec10>best_rec:
                best_rec=rec10
                ndcg = ndcg10
                torch.save(model.state_dict(),'models/TTR_weights.pt')
            print(f'Epoch: {epoch} | Recall@10: {rec10} | Ndcg@10: {ndcg10}')
            #print(f"Epoch: {epoch} | Train_avg_loss: {total_loss/len(dl_tr)} | Test_avg_loss: {test_tloss/len(dl_test)}")



