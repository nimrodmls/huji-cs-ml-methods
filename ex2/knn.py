
import numpy as np
import faiss

class KNNClassifier:
    def __init__(self, k, distance_metric='l2'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.Y_train = None

    def fit(self, X_train, Y_train):
        """
        Update the kNN classifier with the provided training data.

        Parameters:
        - X_train (numpy array) of size (N, d): Training feature vectors.
        - Y_train (numpy array) of size (N,): Corresponding class labels.
        """
        self.X_train = X_train.astype(np.float32)
        self.Y_train = Y_train
        d = self.X_train.shape[1]
        if self.distance_metric == 'l2':
            self.index = faiss.index_factory(d, "Flat", faiss.METRIC_L2)
        elif self.distance_metric == 'l1':
            self.index = faiss.index_factory(d, "Flat", faiss.METRIC_L1)
        else:
            raise NotImplementedError
        pass
        self.index.add(self.X_train)

    def predict(self, X):
        """
        Predict the class labels for the given data.

        Parameters:
        - X (numpy array) of size (M, d): Feature vectors.

        Returns:
        - (numpy array) of size (M,): Predicted class labels.
        """
        # distances is a matrix where each row is of size k, containing the kNN distances.
        # idx are the indices of the kNNs, within the training set X_train
        distances, idx = self.index.search(X, self.k)
        # distances are sorted by increasing distance, as per the faiss documentation.
        # we have no use for the distances at the moment

        # Finding the most common class for each of the data points in the given set
        data_point_classes = []
        for data_point in idx:
            classes = self.Y_train[data_point]
            # Extracting all the unique class values, then finding the most common class value
            # by counting the most common index in the unique values array (via the indices array)
            u_values, indices = np.unique(classes, return_inverse=True)
            majority_class = u_values[np.argmax(np.bincount(indices))]
            data_point_classes.append(majority_class)

        return np.array(data_point_classes)
            
            
        

    def knn_distance(self, X):
        """
        Calculate kNN distances for the given data. You must use the faiss library to compute the distances.
        See lecture slides and https://github.com/facebookresearch/faiss/wiki/Getting-started#in-python-2 for more information.

        Parameters:
        - X (numpy array) of size (M, d): Feature vectors.

        Returns:
        - (numpy array) of size (M, k): kNN distances.
        - (numpy array) of size (M, k): Indices of kNNs.
        """
        X = X.astype(np.float32)
	#### YOUR CODE GOES HERE ####
