#读取数据
from zipfile import ZipFile
import pandas as pd
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
myzip=ZipFile("train.csv.zip")
fp=myzip.open("train.csv")
train_data=pd.read_csv(fp,header=None)
fp.close()
myzip.close()
myzip=ZipFile("test.csv.zip")
fp=myzip.open("test.csv")
test_data=pd.read_csv(fp,header=None)
fp.close()
myzip.close()
#处理数据
import torch as t
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.utils.data as Data
#from IPython import display
class classifer(nn.Module):
    def __init__(self):
        super(classifer, self).__init__()
        self.class_col = nn.Sequential(
            nn.Linear(40,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,23),
        )
    def forward(self, x):
        out = self.class_col(x)
        return out
from torch import optim
model = classifer() # 实例化模型
loss_fn = nn.CrossEntropyLoss() # 定义损失函数
optimiser = optim.SGD(params=model.parameters(), lr=0.05) # 定义优化器
from torch.autograd import Variable
import torch.utils.data as Data
X_train=t.Tensor(np.array(train_data.iloc[:200,:39]))
Y_train = t.Tensor(np.array(train_data.iloc[:200,40]))
X_test = t.Tensor(np.array(test_data))
#X_train=t.Tensor(train_data.iloc[:,:39])
#Y_train = t.Tensor(train_data.iloc[:,40])
#X_test = t.Tensor(test_data)
# 使用batch训练
torch_dataset = Data.TensorDataset(X_train, Y_train) # 合并训练数据和目标数据
MINIBATCH_SIZE = 25
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=MINIBATCH_SIZE,
    shuffle=True,
    num_workers=10           # set multi-work num read data
)
loss_list = []
plt.style.use('ggplot')
for epoch in range(70):
    for step, (batch_x, batch_y) in enumerate(loader):
        optimiser.zero_grad() # 梯度清零
        out = model(batch_x) # 前向传播
        loss = loss_fn(out, batch_y) # 计算损失
        loss.backward() # 反向传播
        optimiser.step() # 随机梯度下降
    loss_list.append(loss)
    if epoch%10==0:
        outputs_train = model(X_train)
        _, predicted_train = t.max(outputs_train, 1)
        outputs_test = model(X_test)
        _, predicted_test = t.max(outputs_test, 1)
        #predicted_test就是训练出的结果
    
        # 同时画出训练集和测试的效果
        #display.clear_output(wait=True)
        #fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13,7))
        #axes[0].scatter(X_train[:,0].numpy(),X_train[:,1].numpy(),c=predicted_train)
        #axes[0].set_xlabel('train')
        #axes[1].scatter(X_test[:,0].numpy(),X_test[:,1].numpy(),c=predicted_test)
        #axes[1].set_xlabel('test')
        #display.display(fig)
print(predicted_test)