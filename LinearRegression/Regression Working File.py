import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
from pathlib import Path

data = pd.read_csv(Path(__file__).parent.joinpath("student-mat.csv"), sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

"""
best = 0
for _ in range(100000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    # Part where trains using the algorithm
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)

    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

print("The best one was:", best)
"""

pickle_in = open(Path(__file__).parent.joinpath("studentmodel.pickle"), "rb")
linear = pickle.load(pickle_in)

print("")
print("Coefficient: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)
print("")

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# Plot a few pieces of information
p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()