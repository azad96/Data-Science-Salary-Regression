import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor

def train_regressor(model, X_train, y_train, X_val, y_val, param_grids={}):
    grid_search = GridSearchCV(model, param_grids, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print(f'Best Parameters: {grid_search.best_params_}')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'R-squared: {r2}')

    return best_model

# https://www.kaggle.com/datasets/saurabhbadole/latest-data-science-job-salaries-2024
df = pd.read_csv('data/DataScience_salaries_2024.csv')

# Make all string columns lowercase and replace spaces with underscores
for column in df.select_dtypes(include=['object']).columns:
    df[column] = df[column].str.lower().str.replace(' ', '_')

df = df[df["employment_type"] == "ft"]
columns = ["work_year", "experience_level", "job_title", "salary_in_usd", "employee_residence", "remote_ratio", "company_location", "company_size"]
df = df[columns]

numerical = ["work_year", "remote_ratio"]
categorical = ["experience_level", "job_title", "employee_residence", "company_location", "company_size"]

# Split the data into training, validation, and test sets
seed = 42
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=seed)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=seed)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = np.log1p(df_train.salary_in_usd.values)
y_val = np.log1p(df_val.salary_in_usd.values)
y_test = np.log1p(df_test.salary_in_usd.values)

del df_train['salary_in_usd']
del df_val['salary_in_usd']
del df_test['salary_in_usd']

# Prepare the data
dv = DictVectorizer(sparse=False)

train_dict = df_train.to_dict(orient='records')
val_dict = df_val.to_dict(orient='records')
test_dict = df_test.to_dict(orient='records')

X_train = dv.fit_transform(train_dict)
X_val = dv.transform(val_dict)
X_test = dv.transform(test_dict)

# ### Random Forest Regression
random_forest_regressor = RandomForestRegressor()
param_grids = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 10, 20]
    }
best_random_forest_regressor = train_regressor(random_forest_regressor, X_train, y_train, X_val, y_val, param_grids)

# ### Testing
y_pred = best_random_forest_regressor.predict(X_test) 

plt.scatter(y_test, y_pred)
plt.xlabel('Ground truth')
plt.ylabel('Predictions')
plt.title('Salary Predictions')
# overlay the regression line
z = np.polyfit(y_test, y_pred, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test), color='magenta')
plt.show()

# ### Saving the Model
model_name = 'model.bin'
with open(model_name, 'wb') as f_out:
  pickle.dump((dv, best_random_forest_regressor), f_out)

print(f'Model is saved to {model_name}')
