import numpy as np

class KNearestNeighbors:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        """Remember train set"""
        self.X_train = X
        self.y_train = y

    def predict(self, X, num_loops=2):
        """
        Predict labels for test data using this classifier.
        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.
        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 2:
            dist = self.compute_distance_two_loops(X)
        elif num_loops == 1:
            dist = self.compute_distances_one_loop(X)
        else:
            dist = self.compute_distance_no_loop(X)

        return self.predict_labels(dist, k=self.k)

    def compute_distance_two_loops(self, X):
        """Compute distance between all points in train and all points in test"""
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dist = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dist[i, j] = np.linalg.norm(self.X_train[j, :] - X[i, :])
        return dist

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.
        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dist = np.zeros((num_test, num_train))
        for i in range(num_test):
            dist[i, :] = np.linalg.norm(self.X_train - X[i, :], axis=1)
        return dist

    def compute_distance_no_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.
        Input / Output: Same as compute_distances_two_loops
        TL;DR
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dist = np.zeros((num_test, num_train))
        print(self.X_train.shape)
        print(X.shape)
        print(dist.shape)
        #dist = X @ self.X_train.T
        return dist


    def predict_labels(self, dist, k):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.
        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
         gives the distance betwen the ith test point and the jth training point.
        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
         test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dist.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            idx = np.argsort(dist[i, :])[:k]
            assert(len(idx) == k)
            closest_y = np.array([self.y_train[label] for label in idx]) #labels of k neighbors
            counts_y = np.bincount(closest_y)
            y_pred[i] = np.argmax(counts_y)
        return y_pred



X_train = np.array([[0.25, 1.5],
                    [0.333, 1.64],
                    [1.67, 0.38],
                    [1.98, 0.4],
                    [1.56, 0.45],
                    [0.64, 1.5]
], float)
y_train = np.array([1, 1, 0, 0, 0, 1], int)

X_test = np.array([[0.5, 0.95], [1.17, 0.34], [2.5, 0.34], [-0.54, 0.32]], float)
kNN = KNearestNeighbors(k=2)
kNN.fit(X_train, y_train)
y_pred = np.array(kNN.predict(X_test, num_loops=2), int) #true
print(y_pred)

y_pred1 = np.array(kNN.predict(X_test, num_loops=1), int)
print(y_pred1)

#y_pred2 = np.array(kNN.predict(X_test, num_loops=0), int)
#print(y_pred2)