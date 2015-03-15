
#On the digits dataset, plot the cross-validation score of a SVC estimator 
#with an linear kernel as a function of parameter C 
#(use a logarithmic grid of points, from 1 to 10).
import numpy as np
from sklearn import cross_validation
from sklearn import datasets, svm
import matplotlib.pyplot as plt
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
C_s=np.logspace(-10,0,10)
kfold = cross_validation.KFold(len(X_digits), n_folds=5)
scores = list()
for valueC in C_s:
     svc = svm.SVC(kernel='linear',C=valueC)
     [svc.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test]) for train, test in kfold]
     score_current=cross_validation.cross_val_score(svc,X_digits,y_digits,cv=kfold,n_jobs=-1)
     scores.append(np.mean(score_current))

plt.figure(1, figsize=(4, 3))#create figure
plt.clf()#clear figure
plt.semilogx(C_s, scores)
#plt.semilogx(C_s, np.array(scores) + np.array(scores_std), 'b--')
#plt.semilogx(C_s, np.array(scores) - np.array(scores_std), 'b--')
locs, labels = plt.yticks()
plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
plt.ylabel('CV score')
plt.xlabel('Parameter C')
plt.ylim(0, 1.1)
plt.show()