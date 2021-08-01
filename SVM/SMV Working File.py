import sklearn
from sklearn import datasets, metrics, svm
from sklearn.neighbors import KNeighborsClassifier

# - Generate hyperplane
# - Find the largest distance between two points in parallel

cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

x = cancer.data  # Artificial data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection\
    .train_test_split(x, y, test_size=0.2)

# print(x_train, y_train)
classes = ['malignant', 'benign']

clf = svm.SVC(kernel="linear", C=2)
# clf = svm.SVC(kernel="poly", degree=2)
# clf = KNeighborsClassifier(n_neighbors=9)

clf.fit(x_train, y_train)

y_prediction = clf.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_prediction)

print(accuracy)

