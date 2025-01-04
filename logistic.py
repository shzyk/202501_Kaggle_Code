import torch
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D 


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


# 计算AIC
def aic_logistic_regression(model, X, y, n):
    y_pred_prob = model.predict_proba(X)[:, 1]
    log_likelihood = np.sum(y * np.log(y_pred_prob))
    aic = -2 * log_likelihood / n + 2 * model.weights.shape[0] / n
    return aic


class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, penalty='l2', C=1.0):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.penalty = penalty  # 可选 'l1' 或 'l2'
        self.C = C  # 正则化强度
        self.weights = None  # 权重初始化
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # 数值稳定性调整
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def loss(self, X, y_one_hot):
        # 计算预测概率
        predictions = self.softmax(np.dot(X, self.weights))
        
        # Cross-Entropy Loss
        loss = -np.mean(np.sum(y_one_hot * np.log(predictions + 1e-9), axis=1))
        
        # 添加正则化项
        if self.penalty == 'l2':
            loss += (self.C / 2) * np.sum(np.square(self.weights))
        elif self.penalty == 'l1':
            loss += self.C * np.sum(np.abs(self.weights))
        
        return loss

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = np.unique(y).shape[0]
        
        # 初始化权重
        self.weights = np.zeros((n_features, n_classes))
        
        # One-Hot 编码目标变量
        y_one_hot = np.eye(n_classes)[y]
        
        # 梯度下降优化
        for i in range(self.max_iter):
            # 预测
            predictions = self.softmax(np.dot(X, self.weights))
            
            # 计算梯度
            dw = np.dot(X.T, (predictions - y_one_hot)) / n_samples
            
            # 添加正则化项的梯度
            if self.penalty == 'l2':
                dw += self.C * self.weights
            elif self.penalty == 'l1':
                dw += self.C * np.sign(self.weights)
            
            # 更新权重
            self.weights -= self.learning_rate * dw
            
            # 打印损失值
            if i % 100 == 0:
                current_loss = self.loss(X, y_one_hot)
                print(f"Iteration {i}: Loss = {current_loss}")
    
    def predict(self, X):
        logits = np.dot(X, self.weights)
        return np.argmax(self.softmax(logits), axis=1)

    def predict_proba(self, X):
        logits = np.dot(X, self.weights)
        return self.softmax(logits)
    

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
    X_train, X_verify, Y_train, Y_verify = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)

    ## 模型训练
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    ## 模型评估
    Y_pred = model.predict(X_verify)
    aic = aic_logistic_regression(model, X_verify, Y_verify, len(Y_train))
    print("The Accuracy is "+str(accuracy_score(Y_verify, Y_pred)*100),"%")
    print("The aic is "+ str(aic))

    # ## t-SNE 可视化 (三维)
    # tsne = TSNE(n_components=3, random_state=seed)
    # X_tsne = tsne.fit_transform(X_verify)
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=Y_verify, cmap='viridis', alpha=0.7)
    # plt.colorbar(scatter)
    # ax.set_title("t-SNE 3D Visualization of Validation Set Predictions")
    # ax.set_xlabel("t-SNE Component 1")
    # ax.set_ylabel("t-SNE Component 2")
    # ax.set_zlabel("t-SNE Component 3")
    # plt.savefig('tsne_3d_visualization.png')  # 保存为PNG文件
    # plt.close()  # 关闭图像以释放内存

    ## t-SNE 可视化
    tsne = TSNE(n_components=2, random_state=seed)
    X_tsne = tsne.fit_transform(X_verify)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y_verify, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title("t-SNE Visualization of Validation Set Predictions")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.savefig('tsne_visualization.png')  # 保存为PNG文件
    plt.close()  # 关闭图像以释放内存

    ## 模型预测
    # df_test = pd.read_csv("../datas/test.csv", header=None)
    # X_test = df_test.iloc[:,:].values
    # X_test = scaler.transform(X_test)
    # X_test = pca.transform(X_test)
    # Y_test = model.predict(X_test)

    # data_out = np.column_stack((np.array(range(0, 10000)), Y_test.astype(int)))
    # df_out = pd.DataFrame(data_out, columns=["Id", "Label"])
    # df_out.to_csv('output.csv', index=False)