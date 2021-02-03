import keras
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Dropout, Lambda, LSTM, Activation, GlobalAveragePooling1D, concatenate, Dense, SpatialDropout1D
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from tensorflow.keras import regularizers
import numpy as np
import random

def base_model(cfg):
    """
    Function for constructing the base model for training deep metrics learning with offline strategy. In offline mining strategy, 
    there are three models for the triplet input (anchor, positive, negative) with shared architecture and weights.
    
    Arguments: 
    cfg - python dict, that contains the configuration. (check Deep_Metrics_Learning.ipynb)    
    
    Returns"
    model - class type variable Keras.Model, the base model object that contains the base model for offline mining.
    """
    
    ip = Input(shape=(cfg['window_width'], 2*cfg['num_of_joints']))

    x = LSTM(units=64, kernel_initializer='glorot_uniform', dropout=cfg['lstm_dropout'])(ip)

    y = Conv1D(64, 8, padding='same', kernel_initializer='glorot_uniform')(ip)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = SpatialDropout1D(cfg['1d_spatial_dropout'])(y)

    y = Conv1D(128, 8, padding='same', kernel_initializer='glorot_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = SpatialDropout1D(cfg['1d_spatial_dropout'])(y)

    y = Conv1D(128, 8, padding='same', kernel_initializer='glorot_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = SpatialDropout1D(cfg['1d_spatial_dropout'])(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(cfg['vec_dim'], activation='relu')(x)

    # Lambda layer that performs l2 normalization on the feature vectors.
    out = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(out)

    model = Model(ip, out)

    return model

def offline_triplet_network(cfg, base_model):
    """
    Function for constructing the triplet model for training deep metrics learning with offline strategy. In offline mining strategy, 
    there are three models for the triplet input (anchor, positive, negative) with shared architecture and weights. This function
    constructs the model which has three inputs for the triplets and outputs three feature vectors that are stacked together.

    Arguments: 
    cfg - python dict, that contains the configuration. (check Deep_Metrics_Learning.ipynb)    
    base_model - class type variable Keras.Model, the base model object that contains the base model for offline mining
    
    Returns"
    model - class type variable Keras.Model, full triplet model object that contains three parallel architecture of the base model for offline mining.
    """
    
    input_shape=(cfg['window_width'], 2 * cfg['num_of_joints'])
    
    # define input: query, positive, negative
    query = Input(shape=input_shape, name="query_input")
    positive = Input(shape=input_shape, name="positive_input")
    negative = Input(shape=input_shape, name="negative_input")

    # construct the base model for the anchor, positive and negative inputs.
    q_vec = base_model(query)
    p_vec = base_model(positive)
    n_vec = base_model(negative)

    # stack outputs - feature vectors for the anchor, positive and negative sequences.
    stacks = Lambda(lambda x: K.stack(x, axis=1), name="output")([q_vec, p_vec, n_vec])

    # Construct the triplet model.
    model = Model(inputs=[query, positive, negative], outputs=stacks, name="triplet_network")

    return model

def online_triplet_network(cfg):
    """
    Function that constructs the model for training deep metrics learning with online mining strategy. In online mining strategy there is only one copy of the
    model. The model takes in batch_size number of sequences as input and outputs batch_size number of feature vectors. From the distances between the feature vectors 
    and the label, triplets are chosen within the mini-batch.
    
    Arguments: 
    cfg - python dict, that contains the configuration. (check Deep_Metrics_Learning.ipynb)    
    
    Returns"
    model - class type variable Keras.Model, triplet model object that contains the architecture for online mining.
    """
    
    ip = Input(shape=(cfg['window_width'], 2 * cfg['num_of_joints']))

    x = LSTM(128)(ip)
    x = Dropout(cfg['normal_dropout'])(x)

    y = Conv1D(64, 8, padding='same', kernel_initializer='glorot_uniform')(ip)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 8, padding='same', kernel_initializer='glorot_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 8, padding='same', kernel_initializer='glorot_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(cfg['vec_dim'], activation='relu')(x)
    
    # Lambda layer that performs l2 normalization on the feature vectors.
    out = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(out)

    model = Model(ip, out)

    return model