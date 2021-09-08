import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
dataset = pd.read_csv('50_Startups.csv')
dataset.drop(['State'],axis=1,inplace=True)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""## Training the Multiple Linear Regression model on the Training set"""

regressor = LinearRegression()
regressor.fit(X_train, y_train)
pickle.dump(regressor,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))
