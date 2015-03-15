from sklearn import cluster, datasets
iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target
X_iris = X_iris[:-10]
y_iris = y_iris[:-10]
import numpy as np
np.random.seed(0)
k_means = cluster.MiniBatchKMeans(n_clusters=3,n_init=1000)
k_means.fit(X_iris) 
print(k_means.predict(X_iris))
print(k_means.labels_[::10])
print(k_means.score(X_iris))
print(y_iris[::10])