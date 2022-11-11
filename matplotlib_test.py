import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv')

df.drop(labels='Species',inplace=True,axis=1)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3,random_state=10)
kmeans.fit(df)
y_pred = kmeans.labels_


distance = []
for i in range(150):
    if y_pred[i]==0:
        distance.append((float)(np.sqrt((df.iloc[i:i+1, [2, 3]] - kmeans.cluster_centers_[0][2:4]) ** 2).sum(axis=1)))
    if y_pred[i]==1:
        distance.append((float)(np.sqrt((df.iloc[i:i+1, [2, 3]] - kmeans.cluster_centers_[1][2:4]) ** 2).sum(axis=1)))
    if y_pred[i]==2:
        distance.append((float)(np.sqrt((df.iloc[i:i+1, [2, 3]] - kmeans.cluster_centers_[0][2:4]) ** 2).sum(axis=1)))

df['y_pred'] = y_pred
df['distance'] = distance


df1 = df.groupby(by='y_pred').apply(lambda x:x.sort_values('distance',ascending=False))
outlier0 = df1[df1.y_pred==0]
outlier0 = outlier0[0:5]

outlier1 = df1[df1.y_pred==1]
outlier1 = outlier1[0:5]

outlier2 = df1[df1.y_pred==2]
outlier2 = outlier2[0:5]


x0 = df[y_pred == 0]
x1 = df[y_pred == 1]
x2 = df[y_pred == 2]

plt.scatter(x0.loc[:,'Petal_Length'],x0.loc[:,'Petal_Width'],c="red",marker='o',label='label0')
plt.scatter(x1.loc[:,'Petal_Length'],x1.loc[:,'Petal_Width'],c="green",marker='*',label='label1')
plt.scatter(x2.loc[:,'Petal_Length'],x2.loc[:,'Petal_Width'],c="blue",marker='+',label='label2')
plt.scatter(kmeans.cluster_centers_[0][2:3],kmeans.cluster_centers_[0][3:4],c="black")
plt.scatter(kmeans.cluster_centers_[1][2:3],kmeans.cluster_centers_[1][3:4],c="black")
plt.scatter(kmeans.cluster_centers_[2][2:3],kmeans.cluster_centers_[2][3:4],c="black")

plt.scatter(outlier0.Petal_Length,outlier0.Petal_Width,c="gray")
plt.scatter(outlier1.Petal_Length,outlier1.Petal_Width,c="gray")
plt.scatter(outlier2.Petal_Length,outlier2.Petal_Width,c="gray")

plt.show()
