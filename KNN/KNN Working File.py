import os
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model, preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle

data = pd.read_csv(Path(__file__).parent.joinpath("car.data"))
print(data.head())

# Transforms the attributes strings in numeric attributes
le = preprocessing.LabelEncoder()

# The distributed number are in alphabetic order.
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))

# high = 0
# low = 1
# med = 2
safety = le.fit_transform(list(data["safety"]))

cls = le.fit_transform(list(data["class"]))
print(lug_boot)

predict = "class"

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.3)

print(x_train, y_train)
model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)
names = ["horrivel", "razoavel", "bom", "supinpa"]

for x in range(len(x_test)):
    # print("Previsto: ", names[predicted[x]], " |  Data (dados): ", x_test[x], " |  Atual: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 9, True)
    # print("N: ", n)
