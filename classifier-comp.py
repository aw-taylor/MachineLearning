import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

X_train = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1], [-5, -3], [4, -2], [2, 6], [-8, 4]])
y_train = np.array([1, 1, 2, 2, 1, 2, 2, 1])
X_test = np.array([[-3, -2], [-1, 1], [3, 2], [5, 0], [6, -1], [-8, 3], [-3, -5], [7, 3]])
y_test = np.array([1, 1, 2, 2, 2, 1, 1, 2])

clf1 = GaussianNB()
clf1.fit(X_train, y_train)
clf1.predict(X_test)
acc1 = clf1.score(X_test, y_test)
print(acc1) # 1.0 at 8 items

clf2 = SVC(kernel='rbf', C=10000.0)
clf2.fit(X_train, y_train)
clf2.predict(X_test)
acc2 = clf2.score(X_test, y_test)
print(acc2) # 1.0 at 4 items

clf3 = DecisionTreeClassifier(min_samples_split=2)
clf3.fit(X_train, y_train)
clf3.predict(X_test)
acc3 = clf3.score(X_test, y_test)
print(acc3) # 1.0 at 6 items

clf4 = KNeighborsClassifier(n_neighbors=3)
clf4.fit(X_train, y_train)
clf4.predict(X_test)
acc4 = clf4.score(X_test, y_test)
print(acc4) # 1.0 at 4 items