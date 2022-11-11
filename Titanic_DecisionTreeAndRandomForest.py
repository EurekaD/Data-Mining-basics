import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("Titanic.csv")

# 了解数据情况
# print(df.shape) (891, 12)

# print(df.columns)
# 'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
# 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'

# print(df.describe())

# 清洗数据
# 查看数据缺失值
# print(df.isnull().sum())
# Age            177   Cabin          687    Embarked         2

# Cabin 船舱 缺失值太多，删除整列特征
df.drop(labels='Cabin', axis=1, inplace=True)

# Embarked 乘船地点 适合众数填充
# print(df["Embarked"].mode())
df["Embarked"].fillna(value='S', inplace=True)

# Age 重要特征 ，这里使用均值填充
df["Age"].fillna(value=df["Age"].mean(), inplace=True)

# 检查缺失值处理
# print(df.isnull().any()) all False

# 取出标签列
y = df["Survived"]

# 删除无用特征和标签列
x = df.drop(
    columns=["Name",
             "Ticket",
             "Survived",
             "PassengerId"]
)

# 分类标签改为数值类型 Sex  Embarked
le = LabelEncoder()
data_sex = le.fit_transform(x.Sex)
data_Emb = le.fit_transform(x.Embarked)
x.Sex = data_sex
x.Embarked = data_Emb

# from sklearn.preprocessing import StandardScaler
# sca = StandardScaler()
# x = sca.fit_transform(x)
# x = pd.DataFrame(x)
# print(x.head())

# 拆分数据集
# print(x.head())
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=6)

# 采用网格搜索算法进行模型参数优化
# from sklearn.model_selection import GridSearchCV
# parameters = {'max_depth': [4,6,7,9],'min_samples_leaf': [4,6,8,10],'min_samples_split':[5,7,9,12]}
# #调用网格搜索模型进行最优化参数搜索
# model_gs = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=parameters, cv=10, n_jobs=4)
# model_gs.fit(X_train,y_train)
# print("优化后准确率: ",model_gs.best_score_)
# print("最佳参数: ",model_gs.best_params_)

dstree = DecisionTreeClassifier(max_depth=7, min_samples_leaf=6, min_samples_split=5)
dstree.fit(X_train,y_train)
print("决策树测试集上的准确率: ",dstree.score(X_test,y_test))
pred = dstree.predict(X_train)
print('决策树训练集上的准确率:',metrics.accuracy_score(y_train,pred))
# from sklearn.tree import export_graphviz
# print(export_graphviz(dstree))

# 绘制roc
# y_score = dstree.predict_proba(X_test)[:,1]
# fpr, tpr, threshold = roc_curve(y_test,y_score)
# roc_auc = metrics.auc(fpr,tpr)
# plt.stackplot(fpr,tpr,color='steelblue',alpha=0.5,edgecolor='black')
# plt.plot(fpr,tpr,color='black',lw=1)
# plt.plot([0,1],[0,1],color='red',linestyle='--')
# plt.text(0.5,0.3,'ROCcurve(area = %0.2f)'%roc_auc)
# plt.xlabel('1-Specificity')
# plt.ylabel('Sensitivity')
# plt.show()

# from sklearn.model_selection import GridSearchCV
# parameters = {'n_estimators':[100,200,250,300],'max_depth': [4,6,7,9],'min_samples_leaf': [2,3,4,6],'min_samples_split':[5,7,9,12]}
# #调用网格搜索模型进行最优化参数搜索
# model_gs = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters, cv=10, n_jobs=4)
# model_gs.fit(X_train,y_train)
# print("优化后准确率: ",model_gs.best_score_)
# print("最佳参数: ",model_gs.best_params_)
# 'max_depth': 6, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100

rdforest = RandomForestClassifier(max_depth=6, min_samples_leaf=2, min_samples_split=5, n_estimators=100)
rdforest.fit(X_train, y_train)
print("随机森林测试集上的准确率: ", rdforest.score(X_test, y_test))
pred = rdforest.predict(X_train)
print('决策树训练集上的准确率:', metrics.accuracy_score(y_train, pred))

# y_score = rdforest.predict_proba(X_test)[:, 1]
# fpr, tpr, threshold = roc_curve(y_test, y_score)
# roc_auc = metrics.auc(fpr, tpr)
# plt.stackplot(fpr, tpr, color='steelblue', alpha=0.5, edgecolor='black')
# plt.plot(fpr, tpr, color='black', lw=1)
# plt.plot([0, 1], [0, 1], color='red', linestyle='--')
# plt.text(0.5, 0.3, 'ROCcurve(area = %0.2f)'%roc_auc)
# plt.xlabel('1-Specificity')
# plt.ylabel('Sensitivity')
# plt.show()

# easy way:
# metrics.plot_roc_curve(rdforest, X_test, y_test)
# plt.show()

importance = rdforest.feature_importances_
Impt_Series = pd.Series(importance, index=X_train.columns)
Impt_Series.sort_values(ascending=True).plot(kind='barh')
plt.show()
