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


import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score

# 选两个模型进行对比 (例如逻辑回归和决策树)
model1 = models["Logistic Regrssion"].fit(X_train, y_train)
model2 = models["Decision Tree"].fit(X_train, y_train)

# 获取概率预测
y_prob1 = model1.predict_proba(X_test)[:, 1]
y_prob2 = model2.predict_proba(X_test)[:, 1]

# 计算P-R曲线和ROC曲线
precision1, recall1, _ = precision_recall_curve(y_test, y_prob1)
precision2, recall2, _ = precision_recall_curve(y_test, y_prob2)

fpr1, tpr1, _ = roc_curve(y_test, y_prob1)
fpr2, tpr2, _ = roc_curve(y_test, y_prob2)

# 计算AUC
auc1 = roc_auc_score(y_test, y_prob1)
auc2 = roc_auc_score(y_test, y_prob2)

# 绘图（同一张图对比）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# P-R曲线
ax1.plot(recall1, precision1, label=f'Logistic Regrssion (AUC={auc1:.3f})')
ax1.plot(recall2, precision2, label=f'Decision Tree (AUC={auc2:.3f})')
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')
ax1.set_title('P-R Curve')
ax1.legend()

# ROC曲线
ax2.plot(fpr1, tpr1, label=f'Logistic Regrssion (AUC={auc1:.3f})')
ax2.plot(fpr2, tpr2, label=f'Decision Tree (AUC={auc2:.3f})')
ax2.plot([0, 1], [0, 1], 'k--')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve')
ax2.legend()

plt.tight_layout()
plt.show()



# 拆解原始测试集的正例和反例
test_pos_idx = (y_test == 1)
test_neg_idx = (y_test == 0)

X_test_pos = X_test[test_pos_idx]
y_test_pos = y_test[test_pos_idx]
X_test_neg = X_test[test_neg_idx]
y_test_neg = y_test[test_neg_idx]

# 正例中只保留10%
sample_ratio = 0.1
num_pos_keep = int(len(y_test_pos) * sample_ratio)
# 随机抽样
np.random.seed(42)
keep_indices = np.random.choice(len(y_test_pos), num_pos_keep, replace=False)

X_test_pos_new = X_test_pos[keep_indices]
y_test_pos_new = y_test_pos[keep_indices]

# 合并构成新的不平衡测试集
X_test_imb = np.vstack((X_test_neg, X_test_pos_new))
y_test_imb = np.hstack((y_test_neg, y_test_pos_new))

# 在极端不平衡测试集上评估最优模型
y_pred_imb = model1.predict(X_test_imb)
y_prob_imb = model1.predict_proba(X_test_imb)[:, 1]

acc_imb = accuracy_score(y_test_imb, y_pred_imb)
auc_imb = roc_auc_score(y_test_imb, y_prob_imb)

print(f"不平衡测试集 (正例仅保留10%) - 精度: {acc_imb:.4f}, AUC: {auc_imb:.4f}")