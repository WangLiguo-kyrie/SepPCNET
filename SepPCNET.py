import torch
import torch.utils.data as Data
from torch.utils.data import Dataset
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data.sampler import WeightedRandomSampler
import sklearn
from sklearn import metrics
import sys

torch.manual_seed(42)

coordinate_collect_train=[]
label_collect_train=[]
with open('./trainset.txt',encoding='utf-8') as f:
    for line in f:
        line.rstrip()
        item=line.split()
        cas=item[0]
        label=item[1]
        coordinate=[]
        with open('./dataset/'+cas+'sparse.txt',encoding='utf-8') as h:
            for hang in h:
                hang.rstrip()
                hangproper=hang.split()
                coordinate.append((float(hangproper[0]),float(hangproper[1]),float(hangproper[2]),float(hangproper[3])))
        coordinate_collect_train.append(coordinate)
        label_collect_train.append(int(label))
        
coordinate_collect_train=np.array(coordinate_collect_train)       
np.savez_compressed('./trainset.npz', arr=coordinate_collect_train)
label_collect_train=np.array(label_collect_train)      
np.savez_compressed('./trainset_label.npz', arr=label_collect_train)
#loading .npz file directly helps to save time 
coordinate=np.load('./trainset.npz')['arr'].astype(np.float32)
print(coordinate.shape)
label=np.load('./trainset_label.npz')['arr'].astype(np.float32)
train_coordinate=torch.from_numpy(coordinate)
train_coordinate=train_coordinate.permute(0,2,1)
train_label=torch.Tensor(label) 
trainset=Data.TensorDataset(train_coordinate,train_label)
BATCH_SIZE_train=64       
weights=[8 if label==1 else 1 for coordinate, label in trainset]
sampler=WeightedRandomSampler(weights,num_samples=1920,replacement=True)
trainset_loader=Data.DataLoader(trainset,batch_size=BATCH_SIZE_train,sampler=sampler,drop_last=True)

coordinate_collect_validation=[]
label_collect_validation=[]
with open('./validationset.txt',encoding='utf-8') as f:
    for line in f:
        line.rstrip()
        item=line.split()
        cas=item[0]
        label=item[1]
        coordinate=[]
        with open('./dataset/'+cas+'sparse.txt',encoding='utf-8') as h:
            for hang in h:
                hang.rstrip()
                hangproper=hang.split()
                coordinate.append((float(hangproper[0]),float(hangproper[1]),float(hangproper[2]),float(hangproper[3])))
        coordinate_collect_validation.append(coordinate)
        label_collect_validation.append(int(label))
        
coordinate_collect_validation=np.array(coordinate_collect_validation)      
np.savez_compressed('./validationset.npz', arr=coordinate_collect_validation)
label_collect_validation=np.array(label_collect_validation)      
np.savez_compressed('./validationset_label.npz', arr=label_collect_validation)
coordinate=np.load('./validationset.npz')['arr'].astype(np.float32)
print(coordinate.shape)
label=np.load('./validationset_label.npz')['arr'].astype(np.float32)
validation_coordinate=torch.from_numpy(coordinate)
validation_coordinate=validation_coordinate.permute(0,2,1)
validation_label=torch.Tensor(label)

   
class CoordinatetoPredict(nn.Module):
    def __init__(self):
        super(CoordinatetoPredict,self).__init__()
        self.conv1=nn.Sequential(nn.Conv1d(4,64,kernel_size=1),nn.BatchNorm1d(64),nn.ReLU(),
                              nn.Conv1d(64,64,kernel_size=1),nn.BatchNorm1d(64),nn.ReLU(),
                              nn.Conv1d(64,128,kernel_size=1),nn.BatchNorm1d(128),nn.ReLU(),
                              nn.Conv1d(128,1024,kernel_size=1),nn.BatchNorm1d(1024),nn.ReLU(),
                              nn.MaxPool1d(kernel_size=4096))

        self.fc2=nn.Sequential(nn.Linear(1024,256),nn.BatchNorm1d(256),nn.ReLU(),
                              nn.Linear(256,64),nn.BatchNorm1d(64),nn.ReLU(),
                              nn.Linear(64,8),nn.BatchNorm1d(8),nn.ReLU(),
                              nn.Linear(8,1),nn.Sigmoid())
                             
    def forward(self, x):
        #print(x.shape)
        x_conv1=self.conv1(x)
        #print(x_conv1.shape)
        x_conv1=x_conv1.view(-1,x_conv1.size(1))
        #print(x_conv1.shape)
        x_fc2=self.fc2(x_conv1)
        #print(x_fc2.shape)
        x_out=x_fc2.view(x_fc2.size(0))
        return x_out

L2=0.001
learning_rate=0.001
model=CoordinatetoPredict()
optimizer=optim.Adam(model.parameters(),weight_decay=L2,lr=learning_rate)
criterion=nn.BCELoss()

with open ('./conv_structure1/minibatch64/{}L2/lr{}/result/trainres.txt'.format(L2,learning_rate),mode='w',encoding="utf-8") as h:
	print('Epoch\tLoss\tAUC',file=h)
with open ('./conv_structure1/minibatch64/{}L2/lr{}/result/validationres.txt'.format(L2,learning_rate),mode='w',encoding="utf-8") as f:
	print('Epoch\tLoss\tAUC',file=f)
AUC_validation_max=0.5
consecutiveepoch_num=0
for epoch in range(60):
    loss_epoch=0
    model.train()
    for batch_data_train in trainset_loader:
        batch_coordinate_train, batch_label_train=batch_data_train
        print(batch_coordinate_train.shape)
        out=model(batch_coordinate_train)
        batch_label_train=batch_label_train.float()
        loss_train=criterion(out,batch_label_train)
        batch_loss_train=loss_train.item()
        loss_epoch=loss_epoch+batch_loss_train*batch_label_train.size(0)
        out_numpy=out.detach().numpy()
        batchlabel_numpy_train=batch_label_train.detach().numpy()
        AUC_train=metrics.roc_auc_score(batchlabel_numpy_train,out_numpy)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
    if (epoch+1) %1 ==0:
        print('*'*10)
        print('epoch {}, Loss {}'.format((epoch+1),(loss_epoch/1920)))	
        print('AUC of Training set: {}'.format(AUC_train))
        with open('./conv_structure1/minibatch64/{}L2/lr{}/result/trainres.txt'.format(L2,learning_rate),mode='a',encoding="utf-8") as h:
            print(str(epoch+1)+'\t'+str(loss_epoch/1920)+'\t'+str(AUC_train),file=h)
            

    model.eval()
    out=model(validation_coordinate)
    validation_label=validation_label.float()
    loss_validation=criterion(out,validation_label)
    batch_loss_validation=loss_validation.item()
    out_numpy=out.detach().numpy()
    batchlabel_numpy_validation=validation_label.detach().numpy()
    AUC_validation=metrics.roc_auc_score(batchlabel_numpy_validation,out_numpy)
    print('*'*10)
    print('epoch {}, Loss {}'.format((epoch+1),(batch_loss_validation)))
    print('AUC of Validation set: {}'.format(AUC_validation))
    if (epoch+1) % 1 ==0:
        with open('./conv_structure1/minibatch64/{}L2/lr{}/result/validationres.txt'.format(L2,learning_rate),mode='a',encoding="utf-8") as f:
            print(str(epoch+1)+'\t'+str(batch_loss_validation)+'\t'+str(AUC_validation),file=f)
    if (epoch+1) % 1 ==0:
        torch.save(model.state_dict(),'./conv_structure1/minibatch64/{}L2/lr{}/ref/train-{:03d}.pth'.format(L2,learning_rate,epoch+1))
    
    if AUC_validation > AUC_validation_max:
        AUC_validation_max=AUC_validation
        consecutiveepoch_num=0
        print('{} consecutive epoches without AUC increase'.format(consecutiveepoch_num))
    else: 
        consecutiveepoch_num+=1
        print('{} consecutive epoches without AUC increase'.format(consecutiveepoch_num))
    
    if consecutiveepoch_num>=15:
        sys.exit(0)
        
        
                  