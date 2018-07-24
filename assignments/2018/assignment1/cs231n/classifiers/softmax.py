import numpy as np
import math
from random import shuffle

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
    
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    # Iterate through all images in batch
    for img in range(num_train):
        
        # Calculate the score of each label
        scores = X[img].dot(W)
        
        # Normalize the scores such that they're all between 0 and 1
        scores -= np.max(scores)
 
        # Find the sum of exponents of the scores
        sum_class_scores = np.sum(np.exp(scores))
                
        loss += np.log(sum_class_scores) - scores[y[img]]
        
        for label in range(num_classes):
            
            dW[:, label] += ( np.exp(scores[label])/sum_class_scores ) * X[img]
            
            if label == y[img]:
                dW[:, label] -= X[img]
        
    
    # Change total loss,grad to average loss,grad per image
    loss /= num_train
    dW /= num_train
    
    # Add loss,grad due to regularization
    loss += reg * np.sum(W * W)
    dW += 2*reg*W
    
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
    
    num_classes = W.shape[1]
    num_train = X.shape[0]
        
    # Calculate the score of each (image, label)
    scores = X.dot(W)
    scores_max = np.max(scores, axis=1)
    scores_norm = (scores.T - scores_max).T
    scores_exp = np.exp(scores_norm)
    scores_sum = np.sum(scores_exp, axis=1)
    scores_frac = ( scores_exp.T/scores_sum ).T
    
    loss = np.sum( np.log(scores_sum) - scores_norm[np.arange(num_train), y] )

    scores_frac[np.arange(num_train), y] -= 1
    dW = np.dot( X.T, scores_frac)

    
    
    # Change total loss,grad to average loss,grad per image
    loss /= num_train
    dW /= num_train
    
    # Add loss,grad due to regularization
    loss += reg * np.sum(W * W)
    dW += 2*reg*W
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

