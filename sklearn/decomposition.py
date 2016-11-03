import numpy as np

class PCA:
    def __init__(self, n_components=None):
        # Store n_components
        self.n_components_ = n_components

        self.components_ = None
        self.explained_variance_ratio_ = []

    def fit(self, x):
        # Use 'x' to learn mean, explained_variance_ratio, and components

        # Create a numpy array
        x = np.array(x)

        # If n_components was not specified in the constructor, then use all components
        if self.n_components_ is None:
            self.n_components_ = x.shape[1]

        # Find the mean of each feature
        self.mean_ = np.mean(x.T, axis=1)

        # NB: np.cov takes the transposed array where rows are variables/features
        covariance = np.cov(x.T)

        # Get the eigenvectors and eigenvalues, use (x^t)(x) to make it a square matrix
        eigValues, eigVectors = np.linalg.eig(covariance)

        # Get indicies that would sort eigValues
        sortIndicies = np.argsort(eigValues)[::-1]

        eigValues = eigValues[sortIndicies]
        eigVectors = eigVectors[:,sortIndicies]

        # Reduce eigVectors to just the most important n_components, based on the largest n eigenvalues
        eigVectors = eigVectors[:, 0:self.n_components_]

        self.components_ = eigVectors.T

        # Calculate explained variance ratios
        totalVariance = eigValues.sum()

        for i in range(self.n_components_):
            self.explained_variance_ratio_.append(eigValues[i]/totalVariance)

        return self

    def transform(self, x):
        # Use mean and components to produce x' from x

        # Create a numpy array
        x = np.array(x)

        # Subtract the mean from each feature
        originCentered = x - self.mean_

        # Multiply by components to rotate/transform data
        return originCentered.dot(self.components_.T)
