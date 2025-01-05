import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
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
        parser.add_argument('-i', '--max_iter', type=int, default=100)

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


class LinearSVM:
    def __init__(self, C=1.0, max_iter=100, learning_rate=0.01):
        self.C = C
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.max_iter):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * (1/self.max_iter) * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * (1/self.max_iter) * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.learning_rate * y_[idx]
    
    def predict(self, X):
        approx = np.dot(X, self.weights) - self.bias
        return np.sign(approx)

class MultiClassSVM:
    def __init__(self, C=1.0, max_iter=100, learning_rate=0.01):
        self.C = C
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.classifiers = {}
    
    def fit(self, X, y):
        unique_classes = np.unique(y)
        for cls in tqdm(unique_classes):
            y_binary = np.where(y == cls, 1, -1)
            svm = LinearSVM(self.C, self.max_iter, self.learning_rate)
            svm.fit(X, y_binary)
            self.classifiers[cls] = svm
    
    def predict(self, X):
        scores = np.zeros((X.shape[0], len(self.classifiers)))
        for i, (cls, svm) in enumerate(self.classifiers.items()):
            scores[:, i] = svm.predict(X)
        return np.argmax(scores, axis=1)


if __name__ == '__main__':


    ## 数据读入
    df = pd.read_csv("../datas/train.csv", header=None)
    X = df.iloc[:,:-1].values
    Y = df.iloc[:,-1].values
    Y = Y.astype(int)


    ## 数据预处理
    X_train, X_verify, Y_train, Y_verify = train_test_split(X, Y, test_size=0.2, random_state=seed+1)
    # 特征归一化
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # 特征降维
    pca = PCA(n_components=0.95)  # 保留95%的方差
    X_train_pca = pca.fit_transform(X_train_scaled)
    print("降维后特征数量：", X_train_pca.shape[1])


    ## 模型训练
    model = MultiClassSVM(max_iter=args.max_iter, learning_rate=0.02)
    model.fit(X_train_pca, Y_train)
    X_verify_scaled = scaler.transform(X_verify)
    X_verify_pca = pca.transform(X_verify_scaled)
    Y_pred = model.predict(X_verify_pca)
    print("The Accuracy is "+str(accuracy_score(Y_verify, Y_pred)*100),"%")


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


    ## 模型选择
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accuracies = []
    for train_index, verify_index in kf.split(X, Y):
        X_train, X_verify = X[train_index], X[verify_index]
        Y_train, Y_verify = Y[train_index], Y[verify_index]

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        pca = PCA(n_components=0.95)  # 保留95%的方差
        X_train_pca = pca.fit_transform(X_train_scaled)
        print("降维后特征数量：", X_train_pca.shape[1])

        model = MultiClassSVM(max_iter=args.max_iter)
        model.fit(X_train_pca, Y_train)
        X_verify_scaled = scaler.transform(X_verify)
        X_verify_pca = pca.transform(X_verify_scaled)
        Y_pred = model.predict(X_verify_pca)
        accuracy = accuracy_score(Y_verify, Y_pred)*100
        print("The Accuracy is " + str(accuracy),"%")
        accuracies.append(accuracy)

    # 输出平均准确率和标准差
    print(f"The average accuracy is {np.mean(accuracies):.2f}%, with a standard deviation of {np.std(accuracies):.2f}%.")
