import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snb
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv('P1_diamonds.csv')

# print(df.head(10).to_string())

# deleting Unnamed column(axis = 1 - column is deleted)
df = df.drop(['Unnamed: 0'], axis=1)

# creating variable for categories
categorical_features = ['cut', 'color', 'clarity']
le = LabelEncoder()

# changing cstegories at numeric valoes
for i in range(3):
    new = le.fit_transform(df[categorical_features[i]])
    df[categorical_features[i]] = new

# print(df.head(10).to_string())
x=df[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']]
y=df[['price']]

# dividing data for test and training packs
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=25, random_state=101)

# training
regr = RandomForestRegressor(n_estimators= 10, max_depth=10, random_state=101)
regr.fit(X_train, y_train.values.ravel())

# prognosing
predictions = regr.predict(X_test)

result = X_test
result['price'] = y_test
result['predictions'] = predictions.tolist()

print(result.to_string())
