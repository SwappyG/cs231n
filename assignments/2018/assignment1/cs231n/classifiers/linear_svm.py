import numpy as np
from random import shuffle

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
    
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        num_loss_increments = 0
        for j in range(num_classes):
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            
            if j == y[i]:
                continue

            if margin > 0:
                num_loss_increments += 1
                dW[:,j] += X[i]
                loss += margin
            
        dW[:,y[i]] -= X[i]*num_loss_increments
            
        
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2*reg*W
  
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
    
    # Determine the number of classes and training images
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    # calculate all scores by matrix multiplying X and W
    scores = X.dot(W)
    
    # Grab the scores of the correct label for each image
    correct_scores = np.choose(y, scores.T)[:, np.newaxis]
    
    # Calculate all the error terms of the loss function in a matrix
    margins = scores - correct_scores + 1
    
    # Create a mask to calculate all gradients at once
    mask = np.zeros(margins.shape)
    
    # mark any indices with margins greater than 1, these add to the loss func
    mask[margins > 0] = 1;
    
    # Count number of margins greater than 0 per image
        # Subtract 1 because we're only interested in the incorrect labels
    loss_count = np.sum(mask, axis=1) - 1
    
    # store the count of num_margins>0 into the correct label index of each image
    mask[np.arange(mask.shape[0]) , y] = -loss_count
    
    # mask[ num_images , num_classes ] now contains coeff of gradient for each class and training image
    # Matrix multiply the mask with the images, divide by num_images for avg, and add reg loss  
    dW = np.dot(X.T,mask) / num_train + 2*reg*W
    
    # Find all margins less than 0 and clamp them to 0
    margins[margins < 0] = 0
    
    # Set all margins for the correct class in each image to 0 as well
    margins[range(scores.shape[0]), y] = 0
    
    # Sum all margins, divide by num_margins for avg, and add reg loss
    loss = np.sum(margins) / num_train + reg * np.sum(W * W)
   
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
