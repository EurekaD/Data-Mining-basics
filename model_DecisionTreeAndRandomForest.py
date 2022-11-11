import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

df = pd.read_csv('model.csv')
# print(df.head(10))
y = df['是否窃漏电']
X = df.drop(labels=['时间', '用户编号', '是否窃漏电'], axis=1)
# print(df.head(10))

# print(df.isnull().any())

# print(df.describe())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=66)

# # 采用网格搜索算法进行模型参数优化
# parameters = {'max_depth': [2, 3, 4], 'min_samples_leaf': [3, 4, 5], 'min_samples_split': [2, 3]}
# #调用网格搜索模型进行最优化参数搜索
# model_gs = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=parameters, cv=5, n_jobs=4)
# model_gs.fit(X_train,y_train)
# print("优化后准确率: ",model_gs.best_score_)
# print("最佳参数: ",model_gs.best_params_)

# 优化后准确率:  0.931183932346723
# 最佳参数:  {'max_depth': 4, 'min_samples_leaf': 4, 'min_samples_split': 5}

# 优化后准确率:  0.9358350951374206
# 最佳参数:  {'max_depth': 3, 'min_samples_leaf': 4, 'min_samples_split': 2}

# 优化后准确率:  0.9358350951374206
# 最佳参数:  {'max_depth': 3, 'min_samples_leaf': 3, 'min_samples_split': 2}

dstree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, min_samples_split=2)
dstree.fit(X_train, y_train)
print("决策树 在测试集上的准确率: ", dstree.score(X_test, y_test))
pred = dstree.predict(X_train)
print('决策树 在训练集上的准确率:', metrics.accuracy_score(y_train, pred))

# 决策树测试集上的准确率:  0.9041095890410958
# 决策树训练集上的准确率: 0.944954128440367


metrics.plot_roc_curve(dstree, X_test, y_test)
plt.show()

y_score = dstree.predict_proba(X_test)[:, 1]
fpr, tpr, threshold = roc_curve(y_test, y_score)
roc_auc = metrics.auc(fpr, tpr)
plt.stackplot(fpr, tpr, color='steelblue', alpha=0.5, edgecolor='black')
plt.plot(fpr, tpr, color='black', lw=1)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.text(0.5, 0.3, 'ROCcurve(area = %0.2f)'%roc_auc)
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.show()

# parameters = {'n_estimators':[50, 100, 150], 'max_depth': [2,3,4], 'min_samples_leaf': [2,3,4], 'min_samples_split':[3,4,5]}
# #调用网格搜索模型进行最优化参数搜索
# model_gs = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters, cv=5, n_jobs=4)
# model_gs.fit(X_train, y_train)
# print("优化后准确率: ", model_gs.best_score_)
# print("最佳参数: ", model_gs.best_params_)

# 优化后准确率:  0.9403805496828752
# 最佳参数:  {'max_depth': 4, 'min_samples_leaf': 3, 'min_samples_split': 5, 'n_estimators': 100}

rdforest = RandomForestClassifier(max_depth=4, min_samples_leaf=3, min_samples_split=5, n_estimators=100)
rdforest.fit(X_train, y_train)
print("随机森林 在测试集上的准确率: ", rdforest.score(X_test, y_test))
pred = rdforest.predict(X_train)
print('随机森林 在训练集上的准确率:', metrics.accuracy_score(y_train, pred))



metrics.plot_roc_curve(rdforest, X_test, y_test)
plt.show()

y_score = rdforest.predict_proba(X_test)[:, 1]
fpr, tpr, threshold = roc_curve(y_test, y_score)
roc_auc = metrics.auc(fpr, tpr)
plt.stackplot(fpr, tpr, color='steelblue', alpha=0.5, edgecolor='black')
plt.plot(fpr, tpr, color='black', lw=1)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.text(0.5, 0.3, 'ROCcurve(area = %0.2f)'%roc_auc)
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.show()