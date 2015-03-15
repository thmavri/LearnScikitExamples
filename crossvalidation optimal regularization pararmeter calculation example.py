from sklearn import cross_validation, datasets, linear_model
import numpy as np
diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
lasso = linear_model.Lasso()
alphas = np.logspace(-4, -.5, 30)
lasso.alphas=alphas
from sklearn.grid_search import GridSearchCV
parameters = {'alpha': alphas}
clf = GridSearchCV(lasso, parameters,n_jobs=-1)
clf.fit(diabetes.data[150:], diabetes.target[150:])
print(clf.best_estimator_.alpha)
print(clf.best_score_)