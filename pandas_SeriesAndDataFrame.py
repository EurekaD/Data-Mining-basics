import pandas as pd
import numpy as np

# （一）创建一个序列
# import random
# array = []
# for _ in range(10):
#     array.append(random.randrange(1,21,1))
# print(array)
# array[8] = np.nan
# series = pd.Series(array)
# print(series)
# print(series.values[1])
# print(series.values[3])
# print(series.values[5])
# print(series.median())
# （二）创建一个数据框
# import random
# dates = pd.Series(pd.date_range(start='10/1/2022', end='10/07/2022'))
# print(dates)
# array = []
# for _ in range(28):
#     array.append(random.random())
# arr = np.array(array).reshape((7,4))
# df = pd.DataFrame(arr,index=dates,columns=['A','B','C','D'])
# print(df)
# # 查看每一列数据的类型
# columns=['A','B','C','D']
# for i in columns:
#     print(i,df[i].dtypes)
# # 4 - 7print(df.columns)
# print(df.index)
# print(df.tail(3))
# print(df.describe())
# # 8 - 10
# df.sort_index(inplace=True,ascending=False)
# print(df)
# df.sort_values(by='A',inplace=True)
# print(df)
# print(df.T)

# （三）数据子集的获取
# import random
# dates = pd.Series(pd.date_range(start='10/1/2022', end='10/07/2022'))
# array = []
# for _ in range(28):
#     array.append(random.random())
# arr = np.array(array).reshape((7,4))
# df = pd.DataFrame(arr,index=dates,columns=['A','B','C','D'])
#
# series = df.loc[ : ,['A']]
# print(type(series))
# print(series)
#
# df = df.reset_index()
# print(df)
# print(df.loc[1:3])
# print(df.iloc[[2,4,5],[1,2]])
# print(df.loc[[1]])
# print(df.loc[:,['A','B']])
# # print(df[df>1]) ????


# （四）数据框的操作
# import random
# dates = pd.Series(pd.date_range(start='10/1/2022', end='10/07/2022'))
# array = []
# for _ in range(28):
#     array.append(random.random())
# arr = np.array(array).reshape((7,4))
# df = pd.DataFrame(arr,index=dates,columns=['A','B','C','D'])
#
# series = pd.Series([1,2,3,4,5,6,7],index=pd.date_range(start='10/2/2022', end='10/08/2022'))
# print(series)
#
# df['E'] = series
# print(df)
# print(df.isnull().sum())
# df.fillna(value=df['E'].mean(),inplace=True)
# print(df)

# （五）应用练习
df = pd.read_csv("titanic.csv")
# print(df.shape)
# print(df.info())
# print((df.head(10)))

df.drop(axis=1, labels=['Name', 'Ticket', 'Cabin'], inplace=True)
print(df.describe())

print(df.isnull().any())
df.fillna(value={'Age': df['Age'].mean(),
                 'Embarked': df['Embarked'].mode()[0]},
          inplace=True)
print(df.isnull().any())

from sklearn import preprocessing

f_names = ['Sex', 'Embarked']
for x in f_names:
    label = preprocessing.LabelEncoder()
    df[x] = label.fit_transform(df[x])
# print(df['Embarked'])

t = pd.pivot_table(data=df, index='Sex', columns='Survived', aggfunc={'Survived': 'count'})
print(t)

t1 = pd.pivot_table(data=df, index=['Survived', 'Sex'], aggfunc={'Age': 'max', 'Pclass': 'count', 'Parch': 'count'})
print(t1)

df.to_csv("titanic_preprocessing.csv")
