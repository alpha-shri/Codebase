import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyploy
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences", "age"]]

predict = "G2"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# Separating Training Data and Test Data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

"""					TRAINING THE MODEL & SAVING IT IN PICKLE, generating a .pickle file

linear = linear_model.LinearRegression()               # Creating our Linear Regression Model

linear.fit(x_train, y_train)                           # Feeding the training set into our Model

accuracy = linear.score(x_test, y_test)

print(accuracy * 100)

with open('studentModel.pickle', 'wb') as f:
	pickle.dump(linear, f)

"""


pickle_in = open('studentModel.pickle', 'rb')
 	
linear = pickle.load(pickle_in)


# print("Co-efficient \n", linear.coef_)
# print("Intercept \n", linear.intercept_)

predictions = linear.predict(x_test)

# for x in range(len(predictions)):
#     print("\n", predictions[x], x_test[x], y_test[x])
 
