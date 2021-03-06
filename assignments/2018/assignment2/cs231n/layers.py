from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    
    N = x.shape[0]
    M = int(np.product(x.shape)/N)
    x_reshaped = np.reshape( x, (N,M) )
    
    out = np.dot(x_reshaped, w) + b
      
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    
    
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    
    x_shape = x.shape
    N = x.shape[0]
    M = int(np.product(x.shape)/N)
    x_reshaped = np.reshape( x, (N,M) )
    
    dw = np.dot(x_reshaped.T, dout)
    dx = np.dot(dout, w.T)
    db = np.sum(dout, axis=0)
    
    dx = np.reshape(dx, x_shape)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    
    out = np.maximum(0, x)
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    
    dx = dout
    dx[x <= 0] = 0 
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        
        # sum value of each feature for all images and divide by num images for avg
        x_sum = np.sum(x, axis=0)
        x_mean = x_sum/N
        
        # find the squared error for each feature in each image
        x_err      = x - x_mean
        x_sq_err   = x_err**2
        
        # sum the squared errors for each feature for all images to find variance
        x_sq_sum   = np.sum(x_sq_err,axis=0)
        x_var      = x_sq_sum/N + eps
        
        # Find standard deviation and the reciprocal of standard deviation
        x_std      = np.sqrt(x_var)
        x_inv_std  = 1./x_std
        
        # subtract mean from each feature in each image and divide by std dev to normalize
        x_norm     = x_err * x_inv_std
        
        # multiply by the scale parameter and add the shift parameter to each feature
        out = gamma * x_norm + beta
        
        # Keep an exponentially decaying running mean and var based on momentum param
        running_mean = momentum*running_mean + (1.0-momentum)*x_mean
        running_var = momentum*running_var + (1.0-momentum)*x_var
        
        # Store all the important stuff in a cache for backward pass
        cache = (x_sum,x_mean,x_err,x_sq_err,x_sq_sum,x_var,x_std,x_inv_std,out,x_norm,x,gamma,beta)
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        
    elif mode == 'test':
        
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        
        # Normalize the input with the running statistics
        x_norm = ( x - running_mean ) / (np.sqrt(running_var + eps))
        
        # Add the learned scale and shift params
        out = x_norm * gamma + beta

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    
    N, D = dout.shape
    
    x_sum,x_mean,x_err,x_sq_err,x_sq_sum,x_var,x_std,x_inv_std,out,x_norm,x,gamma,beta = cache
    
    # Backprop across scaling
    dx_norm      = dout         * gamma
    
    # Backprop multiplication of error and reciprocal of std dev
    dx_err_2     = dx_norm * x_inv_std
    dx_inv_std   = np.sum( dx_norm * x_err , axis=0) #sum since std dev per feature is multiplied for every image
  
    # Backprop through variance and std dev calculation
    dx_std       = dx_inv_std   * -1 * x_std**-2
    dx_var       = dx_std       * 0.5 * x_var**-0.5
    dx_sq_err    = dx_var       * (1/N) * np.ones(x.shape) #reshape due to summation in forward pass
    dx_err_1     = dx_sq_err    * 2 * x_err
    
    # Sum the two branches of the error
    dx_err = dx_err_1 + dx_err_2
    
    # Backprop through the mean calculations
    dx_mean      = np.sum(dx_err, axis=0) # sum since mean per feature is subtracted for every image    
    dx_sum       = (-1./N)*dx_mean*np.ones(x.shape) # Reshape due to summation     
    
    # Sum the two branches for x
    dx = dx_err + dx_sum
    
    # Backprop for gamma and beta
    dgamma = np.sum(dout*x_norm, axis=0) * np.ones_like(gamma)
    dbeta = np.sum(dout,axis=0) 
       
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    
    N, D = dout.shape
    
    x_sum,x_mean,x_err,x_sq_err,x_sq_sum,x_var,x_std,x_inv_std,out,x_norm,x,gamma,beta = cache
    
    dx = gamma*x_inv_std*(1/N) * ( (N*dout) - np.sum(dout,axis=0) - (x_err)*(x_inv_std**2*np.sum(dout*x_err, axis=0)))
     
    dgamma = np.sum(dout*x_norm, axis=0) * np.ones_like(gamma)
    dbeta = np.sum(dout,axis=0)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    
    N, D = x.shape
    
    # sum value of each pixel in each image and divide by num pixels for avg
    # to do this, transpose x to sum in the right direction
    x_sum = np.sum(x.T, axis=0)
    x_mean = x_sum/D

    # find the squared error for each pixel in each image, for all images
    x_err      = x.T - x_mean
    x_sq_err   = x_err**2

    # sum the squared errors for each pixel in each images to find variance
    x_sq_sum   = np.sum(x_sq_err,axis=0)
    x_var      = x_sq_sum/D + eps

    # Find standard deviation and the reciprocal of standard deviation
    x_std      = np.sqrt(x_var)
    x_inv_std  = 1./x_std

    # subtract mean from each pixel in each image and divide by std dev to all images
    x_norm     = x_err * x_inv_std

    # multiply by the scale parameter and add the shift parameter to each feature
    # x_norm needs to be transformed back so each image is a row again instead of column
    out = gamma * x_norm.T + beta

    # Store all the important stuff in a cache for backward pass
    cache = (x_sum,x_mean,x_err,x_sq_err,x_sq_sum,x_var,x_std,x_inv_std,out,x_norm,x,gamma,beta)
    


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    
    N, D = dout.shape
    
    x_sum,x_mean,x_err,x_sq_err,x_sq_sum,x_var,x_std,x_inv_std,out,x_norm,x,gamma,beta = cache
    
    # Backprop across scaling
    # since its normalization per image, and not per feature, transpose dx_norm
    dx_norm      = (dout         * gamma).T
    
    # Backprop multiplication of error and reciprocal of std dev
    dx_err_2     = dx_norm * x_inv_std
    dx_inv_std   = np.sum( dx_norm * x_err , axis=0) #sum since std dev per feature is multiplied for every image
  
    # Backprop through variance and std dev calculation
    # x needs to be transposed since its per image normalization, not per feature
    dx_std       = dx_inv_std   * -1 * x_std**-2
    dx_var       = dx_std       * 0.5 * x_var**-0.5
    dx_sq_err    = dx_var       * (1/D) * np.ones(x.T.shape) #reshape due to summation in forward pass
    dx_err_1     = dx_sq_err    * 2 * x_err
    
    # Sum the two branches of the error
    dx_err = dx_err_1 + dx_err_2
    
    # Backprop through the mean calculations
    # x needs to be transposed since its per image normalization, not per feature
    dx_mean      = np.sum(dx_err, axis=0) # sum since mean per feature is subtracted for every image    
    dx_sum       = (-1./D)*dx_mean*np.ones(x.T.shape) # Reshape due to summation     
    
    # Sum the two branches for x
    # transpose dx to return it such that each row is a new image, and each column is same pixel across images
    dx = (dx_err + dx_sum).T
    
    # Backprop for gamma and beta
    dgamma = np.sum(dout*x_norm.T, axis=0) * np.ones_like(gamma)
    dbeta = np.sum(dout,axis=0)
    
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        
        mask = np.random.choice([0,1], size=x.shape, p=[(1-p),p])
        out = np.multiply(x, mask)/p
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        
        dx = np.multiply(dout, mask) / dropout_param['p']
        
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    pad = conv_param['pad']
    stride = conv_param['stride']
    
    # pad the input along only the H and W dimensions
    x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant', constant_values=0)
    
    # Calculate the output H and W based on padding and stride
    H_prime = np.int(1 + (H + 2 * pad - HH) / stride)
    W_prime = np.int(1 + (H + 2 * pad - HH) / stride)
    
    # initialize the output tensor
    out = np.zeros([N, F, H_prime, W_prime])
   
    # Iterate through all images
    for image in range(0, N):
        # Iterate through all filters
        for filt in range(0, F):
            # Iterate through each row of pixel
            for horz in range(0, H_prime):
                # Iterate through each column of pixel
                for vert in range(0, W_prime):
                    # Grab this filter of size (C, HH, WW)
                    kernel = w[filt,:,:,:]
                    # Grab the sub image of same size as filter
                    sub_image = x_pad[image,:,horz*stride:horz*stride+HH,vert*stride:vert*stride+WW]
                    # Perform convolution (multiply element-wise and sum all products)
                    out[image,filt,horz,vert] = np.sum( kernel * sub_image ) + b[filt]
                            
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    
    x, w, b, conv_param = cache
    
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    
    pad = conv_param['pad']
    stride = conv_param['stride']
    
    # Calculate the output H and W based on padding and stride
    H_prime = np.int(1 + (H + 2 * pad - HH) / stride)
    W_prime = np.int(1 + (H + 2 * pad - HH) / stride)
    
    x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant', constant_values=0)
    
    dx_pad = np.zeros_like(x_pad)
    print(dx_pad.shape)
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    
    # Iterate through all images
    for image in range(0, N):
        # Iterate through all filters
        for filt in range(0, F):
            # Iterate through each row of pixel
            for horz in range(0, H_prime):
                # Iterate through each column of pixel
                for vert in range(0, W_prime):
                    
                    # For every pixel, sub_image is multiplies by kernel
                    # Thus, during backward pass, gradient for sub image is simply the kernel (multiplied by upstream grad)
                    # This adds to any existing gradient acquired by kernels for previous pixels 
                    dx_pad[image,:,horz*stride:horz*stride+HH,vert*stride:vert*stride+WW] += \
                        dout[image, filt, horz, vert] * w[filt, :, :, :]
                    
                    # As above, output is produced by product of sub image and kernel
                    # Thus, during backward pass, gradient for kernel is the sub image
                    # Again, add this for every sub image that uses the same kernel
                    dw[filt,:,:,:] += \
                        dout[image, filt, horz, vert] * x_pad[image,:,horz*stride:horz*stride+HH,vert*stride:vert*stride+WW]
                    
                    # For every kernel, the is just added to the output
                    # Thus, during the backward pass, the gradient is just 1 (multiplied by upstream grad)
                    # Sum this for all pixels, since they all use the same bias
                    db[filt] += dout[image, filt, horz, vert]
    
    # We found the gradient for the padded images, trim the padding to get grad for only the images
    dx = dx_pad[:,:,pad:-pad,pad:-pad]
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    
    N, C, H, W = x.shape
    
    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    stride = pool_param['stride']
    
    H_prime = np.int(1 + (H - pool_h) / stride)
    W_prime = np.int(1 + (W - pool_w) / stride)
    out = np.zeros([N, C, H_prime, W_prime])
    
    # Iterate through all images
    for image in range(0, N):
        # Iterate through each row of pixel
        for horz in range(0, H_prime):
            # Iterate through each column of pixel
            for vert in range(0, W_prime):
                # Grab the sub image of same size as filter
                sub_image = x[image,:,horz*stride:horz*stride+stride,vert*stride:vert*stride+stride]
                # find the maximum value of the sub image, and store it as the output
                out[image,:,horz,vert] = np.amax(sub_image, axis=(1,2))
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    
    x, pool_param = cache
    N,C,H,W = x.shape
    
    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    stride = pool_param['stride']
    
    H_prime = np.int(1 + (H - pool_h) / stride)
    W_prime = np.int(1 + (W - pool_w) / stride)
    dx = np.zeros_like(x)
    
    # Iterate through all images
    for image in range(0, N):
        # Iterate through each row of pixel
        for horz in range(0, H_prime):
            # Iterate through each column of pixel
            for vert in range(0, W_prime):
                
                for chan in range(C):
                    # Grab the sub image of that was pooled for this (horz,vert)
                    sub_image = x[image,chan,horz*stride:horz*stride+stride,vert*stride:vert*stride+stride]
                    
                    # Find the location in sub image that contains the max value
                    pool_mask = (sub_image == np.amax(sub_image))
                    
                    # Multiply the mask by the upstream grad, since only that value is allowed through in max pool
                    # The rest of the grads are all 0
                    dx[image,chan,horz*stride:horz*stride+stride,vert*stride:vert*stride+stride] = \
                        pool_mask * dout[image,chan,horz,vert]
                
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    
    # Get size of each dimension in input
    N, C, H, W = x.shape
    
    # Since we want to normalize across N, H and W, transpose the input
    # Then reshape it down to (N', C), where N' = N*H*W
    x_flat = x.transpose(0,2,3,1).reshape(N*H*W, C)
    
    # With a 2D input, regular batchnorm can be called
    out, cache = batchnorm_forward(x_flat, gamma, beta, bn_param)
    
    # expand out the first dim back into 3 seperate dims ( N,H,W, C)
    # Transpose it so that its back in N,C,H,W order
    out = out.reshape((N,H,W,C)).transpose(0, 3, 1, 2)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    
    # Get size of each dimension in input
    N, C, H, W = dout.shape
    
    # Since we want to normalize across N, H and W, transpose the input
    # Then reshape it down to (N', C), where N' = N*H*W
    dout_flat = dout.transpose(0,2,3,1).reshape(N*H*W, C)
    
    # With a 2D input, regular batchnorm can be called
    dx, dgamma, dbeta = batchnorm_backward(dout_flat, cache)
    
    # expand out the first dim back into 3 seperate dims ( N,H,W, C)
    # Transpose it so that its back in N,C,H,W order
    dx = dx.reshape((N,H,W,C)).transpose(0, 3, 1, 2)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    
    # Grab the dimensions of input
    N,C,H,W = x.shape
    
    # Re-arrange and flatten input to be G by C/G*N*H*W
    x_flat = x.transpose(1,0,2,3).reshape(G, C//G*N*H*W)
    
    ## -- Apply Vanilla batch norm --
    
    # sum value of each pixel in each image and divide by num pixels for avg
    # to do this, transpose x to sum in the right direction
    x_sum = np.sum(x_flat, axis=0)
    x_mean = x_sum/G

    # find the squared error for each pixel in each image, for all images
    x_err      = x_flat - x_mean
    x_sq_err   = x_err**2

    # sum the squared errors for each pixel in each images to find variance
    x_sq_sum   = np.sum(x_sq_err,axis=0)
    x_var      = x_sq_sum/G + eps

    # Find standard deviation and the reciprocal of standard deviation
    x_std      = np.sqrt(x_var)
    x_inv_std  = 1./x_std

    # subtract mean from each pixel in each image and divide by std dev to all images
    x_norm     = x_err * x_inv_std
    
    ## -- end of vanilla batchnorm --
    
    # Reshape and transpose x_norm back to N,C,H,W
    x_norm = x_norm.reshape((C, N, H, W)).transpose(1,0,2,3)
    
    # multiply by the scale parameter and add the shift parameter to each feature
    out = gamma * x_norm + beta

    # Store all the important stuff in a cache for backward pass
    cache = (x_sum,x_mean,x_err,x_sq_err,x_sq_sum,x_var,x_std,x_inv_std,out,x_norm,x,gamma,beta, G)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    
    # get the shape of dout
    N, C, H, W = dout.shape
    
    # grab all the items from the cache
    x_sum,x_mean,x_err,x_sq_err,x_sq_sum,x_var,x_std,x_inv_std,out,x_norm,x,gamma,beta,G = cache    
    
    # Backprop across scaling
    dx_norm      = dout         * gamma
    
    # Reshape x and dx_norm to match forward pass shape
    dx_norm = dx_norm.transpose(1,0,2,3).reshape(G, C//G*N*H*W)
    x = x.transpose(1,0,2,3).reshape(G, C//G*N*H*W)
    
    
    # Backprop multiplication of error and reciprocal of std dev
    dx_err_2     = dx_norm * x_inv_std
    dx_inv_std   = np.sum( dx_norm * x_err , axis=0) #sum since std dev per feature is multiplied for every image
  
    # Backprop through variance and std dev calculation
    # x needs to be transposed since its per image normalization, not per feature
    dx_std       = dx_inv_std   * -1 * x_std**-2
    dx_var       = dx_std       * 0.5 * x_var**-0.5
    dx_sq_err    = dx_var       * (1/G) * np.ones(x.shape) #reshape due to summ in forward pass
    dx_err_1     = dx_sq_err    * 2 * x_err
    
    # Sum the two branches of the error
    dx_err = dx_err_1 + dx_err_2
    
    # Backprop through the mean calculations
    # x needs to be transposed since its per image normalization, not per feature
    dx_mean      = np.sum(dx_err, axis=0) # sum since mean per feature is subtracted for every image    
    dx_sum       = (-1./G)*dx_mean*np.ones(x.shape) # Reshape due to summation     
    
    # Sum the two branches for x
    dx = (dx_err + dx_sum)
    
    # reshape dx back into what x originally was
    dx = dx.reshape((C,N,H,W)).transpose(1,0,2,3)
    
    # Backprop for gamma and beta
    print(np.sum(dout*x_norm, axis=(0,2,3)).shape)
    dgamma = np.sum(dout*x_norm, axis=(0,2,3)) * np.ones_like(gamma)
    dbeta = np.sum(dout,axis=(0,2,3))
    
    
    
    
    
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
