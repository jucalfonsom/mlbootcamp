# K-Nearest Neighbours
# Classify model

from sklearn.neighbors import KNeighborsClassifier

X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

print(knn.predict([[1.1]]))

print(knn.predict_proba([[0.9]]))
