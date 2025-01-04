import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm


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
        parser.add_argument('-m', '--mode', type=str, default="manual", help="test or select")

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


    ## 数据预处理
    df = pd.read_csv("../datas/train.csv", header=None)
    X = df.iloc[:,:-1].values
    Y = df.iloc[:,-1].values
    Y = Y.astype(int)
    # 特征归一化
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    # 特征降维
    pca = PCA(n_components=0.95)  # 保留95%的方差
    X_pca = pca.fit_transform(X_scaled)
    print("降维后特征数量：", X_pca.shape[1])
    
    # 分验证集
    X_train, X_val, Y_train, Y_val = train_test_split(X_pca, Y, test_size=0.2, random_state=seed)
    # 转换为PyTorch张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.long)
    Y_val = torch.tensor(Y_val, dtype=torch.long)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    input_size = X_pca.shape[1]
    hidden_size = 100
    num_classes = 100
    model = MLP(input_size, hidden_size, num_classes)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}], Loss: {loss.item():.4f}')

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


    ## 测试集合
    df_test = pd.read_csv("../datas/test.csv", header=None)
    X_test = df_test.iloc[:,:].values
    X_test = scaler.transform(X_test)
    X_test = pca.transform(X_test)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    test_loader = DataLoader(X_test, batch_size=32, shuffle=False)

    model.eval()
    all_predictions = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.append(predicted.cpu())

    Y_test = torch.cat(all_predictions).numpy()
    data_out = np.column_stack((np.array(range(0, 10000)), Y_test.astype(int)))
    df_out = pd.DataFrame(data_out, columns=["Id", "Label"])
    df_out.to_csv('output.csv', index=False)