import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)       
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
    N, D = X.shape
    C = len(np.unique(y))
    
    for i in range(N):
        y_i = y[i]
        # linear combination : shape  1 * 3073 x 3073 * 10 = 1 * 10
        wx = X[i].dot(W).reshape(1,-1)

        # make softmax stable for numeric issue
        exp_wx = np.exp(wx)

        # softmax : shape : 1 * 10 
        softmax_wx = exp_wx/np.sum(exp_wx, keepdims=True)

        # Loss : shape : scalar
        loss += -np.log(softmax_wx[0,y_i])

        # gradient of loss w.r.t score_i : shape : 1 * 10 
        

        softmax_wx[0,y_i] -= 1

        dW += X[i].reshape(1,-1).T.dot(softmax_wx)
        
        
    loss /= N

    reg_loss =  0.5 * reg * np.sum(W * W)
    loss += reg_loss

    dW /= N
    dW += reg*W
    

    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
    return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)
    
    N, D = X.shape
    C = len(np.unique(y))
    
    y = np.eye(C)[y] # one_hot_y shape N * C
    XW = np.dot(X,W) #shape X : N*D, #shape W : D*C  #XW shape : N * C
    softmax_XW = (np.exp(XW)) / (np.sum(np.exp(XW), axis = 1).reshape(-1,1)) #shape : N*C
    
    loss = -np.log(softmax_XW[np.arange(len(softmax_XW)), #Loss
                              np.argmax(y, axis = 1)]).sum()
    loss /= N
    loss += (reg * (W**2).sum()) / 2
    
    
    softmax_XW[np.arange(len(softmax_XW)), np.argmax(y, axis=1)] -= 1
    
    dW = X.T.dot(softmax_XW) / N
    dW += reg * W
    
    return loss, dW
  # Initialize the loss and gradient to zero.
    

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################