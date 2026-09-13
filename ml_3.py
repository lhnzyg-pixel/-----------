import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
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

cv = StratifiedKFold(n_splits=10,shuffle=True,random_state=42)
result = {}
for name,model in models.items():
    acc_scores = cross_val_score(model,X_train,y_train,cv=cv,scoring='accuracy')
    auc_scores = cross_val_score(model,X_train,y_train,cv=cv,scoring='roc_auc')
    result[name] = {
        "Accurary Mean": acc_scores.mean(),
        "Accurary Std": acc_scores.std(),
        "Auc Mean": auc_scores.mean(),
        "Auc Std": auc_scores.std()
    }

#转换为DataFrame查看结果
result_df = pd.DataFrame(result).T
print(result_df)

from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score

best_model = models["Logistic Regrssion"]
best_model.fit(X_train,y_train)
y_pred = best_model.predict(X_test)

print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))
print(f"精度 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print(f"查准率 (Precision): {precision_score(y_test, y_pred):.4f}")
print(f"查全率 (Recall): {recall_score(y_test, y_pred):.4f}")
print(f"F1值 (F1 Score): {f1_score(y_test, y_pred):.4f}")

