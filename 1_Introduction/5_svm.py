# Support Vector Machines
# Binary classification in its basic form (hyperplane to separeate data on a graph)
# Use of kernels in different dimentions (R^2, R^3)
# You can extend SVM to a multiclass classification

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])

from sklearn.svm import SVC

clf = make_pipeline(StandardScaler(), SVC(gamma="auto"))
clf.fit(X, y)

print(clf.predict([[-5, -5]]))
print(clf.predict([[5, 5]]))
print(clf.predict([[-3, 5]]))
print(clf.predict([[5, -3]]))
print(clf.predict([[0, 0]]))
