import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  n_train, n_feature = X.shape
  n_classes = W.shape[1]
  
  for i in range(n_train):
    X_i = X[i]
    S_i = np.matmul(X_i, W)
    sum_all = np.sum(np.exp(S_i))
    S_yi = S_i[y[i]]
    loss += -np.log(np.exp(S_yi) / sum_all)
    
    top_i = np.exp(S_yi)
    bot_i = sum_all

    for j in range(n_classes):
      term1 = int(j==y[i])
      term2 = np.exp(np.matmul(X[i].T, W[:, j]))/sum_all
      # L_i' = -sum_i[1-exp(xi*w_j)/A]xi, if j==y[i]
      dW[:,j] += (-term1 + term2)*X[i]

  loss /= n_train
  loss += reg * np.sum(W * W)

  dW /= n_train
  dW += 2.*reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  n_train, n_feature = X.shape
  n_classes = W.shape[1]

  # Loss
  score = np.exp(np.matmul(X, W))
  Li = -np.log(score[range(n_train), y]/np.sum(score, axis=1))
  loss += np.sum(Li)
  loss = loss/n_train + reg * np.sum(W * W)
  # Grad
  term1_2 = score/np.sum(score, axis=1)[:,None]
  term1_2[range(n_train), y] = term1_2[range(n_train), y] - 1
  dW = np.matmul(X.T, term1_2)
  dW = dW/n_train + 2.* reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

