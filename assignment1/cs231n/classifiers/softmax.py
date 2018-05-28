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
  class_nums = W.shape[1]
  train_nums = X.shape[0]
  for i in range(train_nums):
    scores = X[i].dot(W)
    expScore = np.exp(scores)
    #loss += -1 * np.log(expScore[y[i]] / np.sum(expScore))
    sumScore = 0.0
    for j in range(class_nums):
        if j==y[i]:
            dW[:,j] += ((expScore[j]/np.sum(expScore)) - 1) * X[i]
        else:
            dW[:,j] += (expScore[j]/np.sum(expScore)) * X[i]
        sumScore += expScore[j]
    loss += -1 * np.log(expScore[y[i]] / sumScore)
  loss = loss/train_nums + reg * np.sum(W*W)
  dW = dW/train_nums + 2*reg*W
            
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
  class_nums = W.shape[1]
  train_nums = X.shape[0]
  expScores = np.exp(X.dot(W))
  xNums = range(train_nums)
  yExpScores = expScores[xNums,y]
  sumExpScores = np.sum(expScores, axis=1)
  loss = np.sum(-np.log(yExpScores / sumExpScores)) / train_nums + reg*np.sum(W*W)
  trainWclass = np.zeros((train_nums, X.shape[1], class_nums))
  for i in range(train_nums):
    for j in range(class_nums):
        if j==y[i]:
            trainWclass[i, :, j] = (expScores[i, y[i]]/sumExpScores[i] - 1)*X[i]#.reshape(X.shape[1], 1)
        else:
            trainWclass[i, :, j] = (expScores[i, j]/sumExpScores[i])*X[i]#.reshape(X.shape[1], 1)
  dW += np.sum(trainWclass,axis=0)/train_nums + 2*reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
    

  return loss, dW
