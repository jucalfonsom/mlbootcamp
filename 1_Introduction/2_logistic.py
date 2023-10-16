# Logistic Regression
# Binary classification (0 or 1)

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
clf = LogisticRegression(random_state=0).fit(X, y)
print(clf.predict(X[:2, :]))

print(clf.predict_proba(X[:2, :]))

print(clf.score(X, y))
