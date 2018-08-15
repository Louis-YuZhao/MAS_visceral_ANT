import numpy as np
from sklearn.neighbors import NearestNeighbors

# version 3
# 2017-06-04
# author louis

# (2) using sklearn

  #%%
class KNearestNeighbor(object):
  """ a kNN classifier """

  def __init__(self, algorithm='auto', metric='minkowski', p=2):
      self.algorithm = algorithm
      self.metric = metric
      self.p = p
      """Parameter for the Minkowski metric from sklearn.metrics.pairwise.pairwise_distances. 
      When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance 
      (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used."""

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,M) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, x_test, k=1):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm=self.algorithm, metric=self.metric, p=self.p,).fit(self.X_train)
    distances, indices = nbrs.kneighbors(x_test)    

    return self.predict_labels(indices, k=k)        
        
  def predict_labels(self, indices, k=1):
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
    
    N, M = np.shape(self.y_train)
    num_test = indices.shape[0]
    y_pred = np.zeros((M,num_test))
    dist_dice =indices[:,0:k]
    for i in xrange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.     
      dice_i = dist_dice[i,:]
      dice_i = list(dice_i.reshape(-1))
      closest_y = self.y_train[dice_i]
      y_pred_temp = np.mean(closest_y, axis=0)      
      y_pred[:,i] = y_pred_temp
            
    return y_pred, indices



