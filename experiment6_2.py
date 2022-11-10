import pandas as pd

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

df = pd.read_excel("NHANES.xlsx")

# print(df.isnull().sum())
# print(df.columns)
# 'age_months', 'sex', 'black', 'BMI', 'HDL', 'CKD_stage', 'S_Creat',
# 'cal_creat', 'meals_not_home', 'CKD_epi_eGFR'

predictors = df.columns[:-1]
# print(predictors)
X_train, X_test, y_train, y_test = train_test_split(df[predictors], df.CKD_epi_eGFR, test_size=0.3, random_state=6)

# max_depth = [18, 19, 20, 21, 22]
# min_samples_split = [2, 4, 6, 8]
# min_samples_leaf = [2, 4, 8, 10, 12]
# parameters = {'max_depth':max_depth, 'min_samples_split':min_samples_split, 'min_samples_leaf':min_samples_leaf}
#
# grid_dtcateg = GridSearchCV(estimator=DecisionTreeRegressor(), param_grid=parameters, cv=10)
# grid_dtcateg.fit(X_train, y_train)
# print(grid_dtcateg.best_params_)

cart_reg = DecisionTreeRegressor(max_depth= 18, min_samples_leaf=2, min_samples_split=2)
cart_reg.fit(X_train,y_train)
pred = cart_reg.predict(X_test)
print(metrics.mean_squared_error(y_test, pred))

rf = RandomForestRegressor(n_estimators=200, random_state=6)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print(metrics.mean_squared_error(y_test, rf_pred))

importance = rf.feature_importances_
Impt_Series = pd.Series(importance, index=X_train.columns)
Impt_Series.sort_values(ascending=True).plot(kind='barh')
plt.show()