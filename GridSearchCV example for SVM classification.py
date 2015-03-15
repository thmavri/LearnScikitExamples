from sklearn import datasets, svm
import numpy as np
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
from sklearn.grid_search import GridSearchCV
gammas = np.logspace(-6, -1, 10)
parameters = {'kernel':('linear', 'rbf'), 'C':[0.001, 1]}
svc = svm.SVC(kernel='linear')
clf = GridSearchCV(svc, parameters,n_jobs=-1)
clf.fit(X_digits[:1000], y_digits[:1000]) 
print(clf.best_score_)
print(clf.best_estimator_.C)
print(clf.best_estimator_.kernel)
print(clf.score(X_digits[1000:], y_digits[1000:]))