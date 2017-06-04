import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, y[i]] -= X[i]
        dW[:, j] += X[i]

  dW /= num_train
  dW += 2.*reg*W

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  n_features, n_classes = W.shape
  n_train = X.shape[0]

  S = np.matmul(X, W) # score
  # S_yi = S[range(n_train), y] # or use np.choose
  S_yi = np.choose(y, S.T) # S_yi

  Li = S - S_yi[:, None] + 1
  Li[range(n_train), y] = 0 # since we are not supposed to subtract S_yi at index yi
  Li[Li < 0] = 0 # take the max
  Li_sum = np.sum(Li, axis=1)

  loss = np.mean(Li_sum) + reg*np.sum(W*W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  mask = np.zeros((n_train, n_classes))
  mask[Li > 0] = 1 # we only count for non-zero elements in Li, which is max(0, S_j-S_yi+1)

  # however, we didn't include these -S_yi yet in Li, it should appear in the calculation
  # the same number of times as non-zero Li, except the sign is negative. 
  nonzero_counts = np.sum(mask, axis=1)
  mask[range(n_train), y] = -1 * nonzero_counts
  
  # now use mask to obtain dW: (D x C)
  dW = np.matmul(X.T, mask)
  # print('!!!!!!!!!!!!!!', n_train)
  dW /= n_train
  dW += 2.*reg*W



  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
