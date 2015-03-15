#Try classifying classes 1 and 2 from the iris dataset with SVMs, with the 2 first features. Leave out 10% of each class and test prediction performance on these observations.
#Warning: the classes are ordered, do not leave out the last 10%, you would be testing on only one class.
#Hint: You can use the decision_function method on a grid to get intuitions.



import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm

iris = datasets.load_iris()
X = iris.data
y = iris.target

X = X[y != 0, :2]
y = y[y != 0]

np.random.seed(0)
lengthX = len(X)
indices = np.random.permutation(lengthX )
X = X[indices]
y = y[indices]

X_train = X[:.9 * lengthX ]
y_train = y[:.9 * lengthX ]
X_test = X[.9 * lengthX :]
y_test = y[.9 * lengthX :]

# fit the model
for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
    clf = svm.SVC(kernel=kernel,C=0.1,gamma=10)
    clf.fit(X_train, y_train)

    print(clf.predict(X_test))
    print(clf.score(X_test,y_test))

    plt.figure(fig_num)
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], zorder=100, cmap=plt.cm.Blues)

    # Circle out the test data
    plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none', zorder=100)

    plt.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
    print(Z)
    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.PuRd)
    plt.contour(XX, YY, Z, colors=['b', 'g', 'r'], linestyles=['--', '-', '--'],
                levels=[-1, 0, 1])

    plt.title(kernel)
plt.show()