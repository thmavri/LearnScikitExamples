from sklearn import linear_model
import numpy as np
from sklearn import datasets

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
np.random.seed(0)
indices_X = np.random.permutation(len(X_digits))
numX = len(X_digits)/10
print numX
X_train = X_digits[indices_X[:-numX]]
y_train = y_digits[indices_X[:-numX]]
X_test = X_digits[indices_X[-numX:]]
y_test = y_digits[indices_X[-numX:]]
regr = linear_model.LogisticRegression(penalty='l2', dual=False, tol=1e-4, C=1000, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None)
regr.fit(X_train, y_train)
print(regr.predict(X_test))
print(regr.score(X_test, y_test, sample_weight=None))