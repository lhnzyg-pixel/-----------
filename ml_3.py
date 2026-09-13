import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

#加载数据
data = load_breast_cancer()
X,y = data.data, data.target

#划分train
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3,stratify=y,random_state=42)

print(f"train 大小：{X_train.shape}",f"test:{X_test.shape}")

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score,StratifiedKFold

models = {
    "Logistic Regrssion":LogisticRegression(max_iter=5000,random_state=42),
    "Decision Tree":DecisionTreeClassifier(max_depth=5,random_state=42),
    "KNN":make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=5))
}