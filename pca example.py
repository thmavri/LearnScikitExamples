# Create a signal with only 2 useful dimensions
import numpy as np
x1 = np.random.normal(size=100)
x2 = np.random.normal(size=100)
x3 = x1 + x2
X = np.c_[x1, x2, x3]

from sklearn import decomposition
pca = decomposition.PCA()
pca.fit(X)
cnt=0
for value in pca.explained_variance_ratio_ :
	if value<0.01 :
	    print value
	    cnt=cnt+1
print cnt
pca.n_components=cnt
print(pca.fit(X))
print('=====Transformed========')
X_reduced=pca.fit_transform(X)
print(X_reduced)

