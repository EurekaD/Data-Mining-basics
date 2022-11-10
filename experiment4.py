import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("titanic_preprocessing.csv")
# print(df.isnull().any())


# 饼图
# male 1 ,female 0
# male = df['Sex'].loc[df['Sex']==1].count()
# female = df['Sex'].loc[df['Sex']==0].count()
#
# Sex_count = [male,female]
# labels = ['Male','Female']
#
# plt.axes(aspect='equal')
# plt.pie(x=Sex_count,
#         labels=labels,
#         autopct='%.lf%%')
# plt.title('Rate of Male and Female')
# plt.show()
#
# plt.axes(aspect='equal')
# plt.pie(x=Sex_count,
#         colors=['blue','red'],
#         explode=[0,0.1],
#         labels=labels,
#         autopct='%.lf%%',
#         wedgeprops={'linewidth':5,'edgecolor':'black'})
# plt.title('Rate of Male and Female')
# plt.show()

# 条形图
# Pclass 1 2 3 三个等级

# Pclass_count = []
# for i in np.arange(1,4):
#     Pclass_count.append(df['Pclass'].loc[df['Pclass']==i].count())
# PclassValueCouunt = pd.DataFrame({'Pclass':['1','2','3'],'Count':Pclass_count})
# # 'Pclass':['1','2','3'] 里面不能是数值类型
#
#
# plt.bar(height=PclassValueCouunt['Count'],
#         x=PclassValueCouunt['Pclass'])
# plt.ylabel('number of people')
# plt.xlabel('Pclass')
# plt.show()
#
#
# PclassValueCouunt.sort_values(by='Count',inplace=True)
# plt.barh(width=PclassValueCouunt['Count'],
#          y=PclassValueCouunt['Pclass'],
#          tick_label=PclassValueCouunt['Pclass'])
# plt.xlabel('number of people')
# plt.ylabel('Pclass')
# plt.show()
#
# # survived
# Pclass_survived = []
# df_survived = df.loc[df['Survived']==1]
# for i in np.arange(1,4):
#     Pclass_survived.append(df_survived['Pclass'].loc[df_survived['Pclass']==i].count())
#
# # dead
# Pclass_dead = []
# df_dead = df.loc[df['Survived']==0]
# for i in np.arange(1,4):
#     Pclass_dead.append(df_dead['Pclass'].loc[df_dead['Pclass']==i].count())
#
#
# plt.bar(height=Pclass_survived,
#         x=np.arange(1,4),
#         color='green',
#         label='survived')
# plt.bar(height=Pclass_dead,
#         x=np.arange(1,4),
#         bottom=Pclass_survived,
#         color='red',
#         label='dead')
# plt.ylabel('number of people')
# plt.xlabel('Pclass')
# plt.legend()
# plt.show()
#
#
# plt.bar(height=Pclass_survived,
#         x=np.arange(1,4),
#         label='survived',
#         color='green',
#         width=0.4)
# plt.bar(height=Pclass_dead,
#         x=np.arange(1,4)+0.4,
#         label='dead',
#         color='red',
#         width=0.4)
# plt.xticks(np.arange(1,4)+0.2,np.arange(1,4))
# plt.ylabel('number of people')
# plt.xlabel('Pclass')
# plt.legend()
# plt.show()

# 直方图
# 使用原始数据集
# titanic = pd.read_csv("titanic.csv")
# print(titanic.isnull().any())
# titanic['Age'].dropna(inplace=True)
# plt.hist(x=titanic.Age,
#          bins=50,
#          color='steelblue',
#          edgecolor='black')
# plt.xlabel('Age')
# plt.ylabel('Number')
# plt.show()


# 箱线图
# titanic = pd.read_csv("titanic.csv")
# # print(titanic.isnull().any())
# titanic['Age'].dropna(inplace=True)
# plt.boxplot(x=titanic.Age,
#             patch_artist=True,
#             showmeans=True,
#             boxprops={'color':'black','facecolor':'steelblue'},
#             flierprops={'marker':'o','markerfacecolor':'red','markersize':3},
#             meanprops={'marker':'D','markerfacecolor':'indianred','markersize':4},
#             medianprops={'linestyle':'--','color':'orange'},
#             labels=[''])
# plt.show()
#
# list = []
# for i in range(2):
#     list.append(titanic.Age[titanic.Survived==i])
# # print(list)
# plt.boxplot(x=list,
#             patch_artist=True,
#             showmeans=True,
#             boxprops={'color':'black','facecolor':'steelblue'},
#             flierprops={'marker':'o','markerfacecolor':'red','markersize':3},
#             meanprops={'marker':'D','markerfacecolor':'indianred','markersize':4},
#             medianprops={'linestyle':'--','color':'orange'},
#             labels=['dead','Survived'])
# plt.show()

plt.figure(figsize=(12,6))

ax1 = plt.subplot2grid(shape=(2,3),loc=(0,0))
male = df['Sex'].loc[df['Sex']==1].count()
female = df['Sex'].loc[df['Sex']==0].count()
Sex_count = [male,female]
labels = ['Male','Female']
plt.pie(x=Sex_count,
        labels=labels,
        autopct='%.lf%%')


ax2 = plt.subplot2grid(shape=(2,3),loc=(0,1))
Pclass_count = []
for i in np.arange(1,4):
    Pclass_count.append(df['Pclass'].loc[df['Pclass']==i].count())
PclassValueCouunt = pd.DataFrame({'Pclass':['1','2','3'],'Count':Pclass_count})
# 'Pclass':['1','2','3'] 里面不能是数值类型
plt.bar(height=PclassValueCouunt['Count'],
        x=PclassValueCouunt['Pclass'])
plt.ylabel('number of people')
plt.xlabel('Pclass')

ax3 = plt.subplot2grid(shape=(2,3),loc=(1,0),colspan=2)
titanic = pd.read_csv("titanic.csv")
print(titanic.isnull().any())
titanic['Age'].dropna(inplace=True)
plt.hist(x=titanic.Age,
         bins=50,
         color='steelblue',
         edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Number')

ax4 = plt.subplot2grid(shape=(2,3),loc=(0,2),rowspan=2)
titanic = pd.read_csv("titanic.csv")
# print(titanic.isnull().any())
titanic['Age'].dropna(inplace=True)
plt.boxplot(x=titanic.Age,
            patch_artist=True,
            showmeans=True,
            boxprops={'color':'black','facecolor':'steelblue'},
            flierprops={'marker':'o','markerfacecolor':'red','markersize':3},
            meanprops={'marker':'D','markerfacecolor':'indianred','markersize':4},
            medianprops={'linestyle':'--','color':'orange'},
            labels=[''])

plt.subplots_adjust(hspace=0.6,wspace=0.3)
plt.show()