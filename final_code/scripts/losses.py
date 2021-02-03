import keras.backend as K

def offline_lossless_triplet_loss(y_true, y_pred):
    """
    Implementation of Lossless triplet loss function, a variant of the triplet loss which introduces log to the loss.

    Arguments: 
    y_true - true labels, required when you define a custom loss in Keras, you don't need it in this function.
    y_pred - python list containing three objects:
             anchor -- the embedding for the anchor data.
             positive -- the embedding for the positive data (similar to anchor).
             negative -- the embedding for the negative data (different from anchor).
    N  - Feature vector dimension.
    beta - The scaling factor, N is recommended.
    epsilon - The Epsilon value to prevent ln(0).

    Returns:
    loss - real number, value of the loss.

    """
    # define constants
    max_dist = K.constant(2 * 2)
    epsilon = K.epsilon()
    beta = max_dist
    zero = K.constant(0.0)

    # get the prediction vector
    query, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]

    # compute euclidean distance
    pos_distance = K.sum(K.square(query - positive), axis=1)
    neg_distance = K.sum(K.square(query - negative), axis=1)

    # non linear values
    # -ln(-x/N+1)
    pos_dist = -K.log(-((pos_distance) / beta) + 1 + epsilon)
    neg_dist = -K.log(-((max_dist - neg_distance) / beta) + 1 + epsilon)

    # compute loss
    partial_loss = pos_dist + neg_dist
    loss = K.mean(K.maximum(partial_loss, zero), axis=0)

    return loss

def offline_triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of triplet loss function while using offline triplet mining strategy. The definition of the triplet loss function
    is explained in this blog, https://omoindrot.github.io/triplet-loss

    Arguments:
    y_true - true labels, required when you define a custom loss in Keras.
    y_pred - python list containing three objects:
             anchor -- the embedding for the anchor data.
             positive -- the embedding for the positive data (similar to anchor).
             negative -- the embedding for the negative data (different from anchor).
    alpha -  floating point value, the margin parameter in the triplet loss function.
    
    Returns:
    loss - real number, value of the loss.
    """    
    
    # get the prediction vector
    query, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]

    # compute distance
    pos_distance = K.sum(K.square(query - positive), axis=1)
    neg_distance = K.sum(K.square(query - negative), axis=1)

    # compute loss
    basic_loss = pos_distance - neg_distance + alpha
    loss = K.mean(K.maximum(basic_loss, 0), axis=0)

    return loss
