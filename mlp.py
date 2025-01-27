import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.manifold import TSNE

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='process.log',
    filemode='w' # a or w
    )
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


import argparse
class Args:
    def parseargs(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-e', '--epoch', type=int, default=50)
        parser.add_argument('-b', '--batch', type=int, default=100)
        parser.add_argument('-l', '--lr', type=float, default=0.001)

        self.pargs = parser.parse_args()
        for key, value in vars(self.pargs).items():
            setattr(self, key, value)

    def __init__(self) -> None:
        self.parseargs()
args = Args()


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

if __name__ == '__main__':


    ## 数据读入
    df = pd.read_csv("../datas/train.csv", header=None)
    X = df.iloc[:,:-1].values
    Y = df.iloc[:,-1].values
    Y = Y.astype(int)


    # ## 数据预处理
    # X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=seed+1)
    # # 特征归一化
    # scaler = MinMaxScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_val_scaled = scaler.transform(X_val)
    # # 特征降维
    # pca = PCA(n_components=0.95)  # 保留95%的方差
    # X_train_pca = pca.fit_transform(X_train_scaled)
    # X_val_pca = pca.transform(X_val_scaled)
    # print("降维后特征数量：", X_train_pca.shape[1])
    

    # ## 模型训练
    # X_train_pca = torch.tensor(X_train_pca, dtype=torch.float32)
    # Y_train = torch.tensor(Y_train, dtype=torch.long)
    # X_val_pca = torch.tensor(X_val_pca, dtype=torch.float32)
    # Y_val = torch.tensor(Y_val, dtype=torch.long)
    # # 创建数据加载器
    # train_dataset = TensorDataset(X_train_pca, Y_train)
    # val_dataset = TensorDataset(X_val_pca, Y_val)
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    # # 创建模型
    # input_size = X_train_pca.shape[1]
    # hidden_size = 100
    # num_classes = 100
    # model = MLP(input_size, hidden_size, num_classes)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # # 训练
    # num_epochs = 50
    # for epoch in tqdm(range(num_epochs)):
    #     model.train()
    #     for inputs, labels in train_loader:
    #         inputs, labels = inputs.to(device), labels.to(device)
            
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
            
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     print(f'Epoch [{epoch+1}], Loss: {loss.item():.4f}')
    # # 验证
    # model.eval()
    # yval = []
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for inputs, labels in val_loader:
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         outputs = model(inputs)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #         yval.append(predicted)
        
    #     yval = torch.cat(yval)
    #     accuracy = 100 * correct / total
    #     print(f'Accuracy of the model on the val set: {accuracy:.2f}%')

    # ## t-SNE 可视化
    # tsne = TSNE(n_components=2, random_state=seed)
    # X_tsne = tsne.fit_transform(X_val)
    # plt.figure(figsize=(8, 6))
    # scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=yval.cpu().numpy(), cmap='viridis', alpha=0.7)
    # plt.colorbar(scatter)
    # plt.title("t-SNE Visualization of Validation Set Predictions")
    # plt.xlabel("t-SNE Component 1")
    # plt.ylabel("t-SNE Component 2")
    # plt.savefig('tsne_visualization.png')  # 保存为PNG文件
    # plt.close()  # 关闭图像以释放内存




    # ## 测试集合
    # df_test = pd.read_csv("../datas/test.csv", header=None)
    # X_test = df_test.iloc[:,:].values
    # X_test = scaler.transform(X_test)
    # X_test = pca.transform(X_test)
    # X_test = torch.tensor(X_test, dtype=torch.float32)
    # test_loader = DataLoader(X_test, batch_size=32, shuffle=False)

    # model.eval()
    # all_predictions = []
    # with torch.no_grad():
    #     for inputs in test_loader:
    #         inputs = inputs.to(device)
    #         outputs = model(inputs)
    #         _, predicted = torch.max(outputs.data, 1)
    #         all_predictions.append(predicted.cpu())

    # Y_test = torch.cat(all_predictions).numpy()
    # data_out = np.column_stack((np.array(range(0, 10000)), Y_test.astype(int)))
    # df_out = pd.DataFrame(data_out, columns=["Id", "Label"])
    # df_out.to_csv('output.csv', index=False)



    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accuracies = []
    for train_index, verify_index in kf.split(X, Y):
        X_train, X_val = X[train_index], X[verify_index]
        Y_train, Y_val = Y[train_index], Y[verify_index]

        # 特征归一化
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        # 特征降维
        pca = PCA(n_components=0.95)  # 保留95%的方差
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_val_pca = pca.transform(X_val_scaled)
        print("降维后特征数量：", X_train_pca.shape[1])
    
        ## 模型训练
        X_train_pca = torch.tensor(X_train_pca, dtype=torch.float32)
        Y_train = torch.tensor(Y_train, dtype=torch.long)
        X_val_pca = torch.tensor(X_val_pca, dtype=torch.float32)
        Y_val = torch.tensor(Y_val, dtype=torch.long)
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_pca, Y_train)
        val_dataset = TensorDataset(X_val_pca, Y_val)
        train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
        # 创建模型
        input_size = X_train_pca.shape[1]
        hidden_size = 100
        num_classes = 100
        model = MLP(input_size, hidden_size, num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # 训练
        num_epochs = args.epoch
        for epoch in tqdm(range(num_epochs)):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # print(f'Epoch [{epoch+1}], Loss: {loss.item():.4f}')
        # 验证
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            print(f'Accuracy of the model on the val set: {accuracy:.2f}%')
        accuracies.append(accuracy)

    # 输出平均准确率和标准差
    print(f"The average accuracy is {np.mean(accuracies):.2f}%, with a standard deviation of {np.std(accuracies):.2f}%.")
