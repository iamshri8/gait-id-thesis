from annoy import AnnoyIndex
import numpy as np

def get_image_features(cfg, model, inp):
    """ 
    This function generates the feature vectors for a single gait sequence during model inference. Note that the function name has the word image in it because, 
    in general terms  of deep metrics learning usually images are the input and hence it is name as get_image_features. In this actual context it is get_sequence_features.
    
    Arguments:
    cfg - python dict, that contains the configuration. (check Deep_Metrics_Learning.ipynb)
    model - Keras.Model variable, triplet model object that has to be used for generating the feature vectors during model inference.
    inp - numpy array, that contains the human joint coordinates of a single gait sequence (input for model inference).
    
    Returns:
    feature_vec - numpy array, that contains the feature vector for the input gait sequence.
    """    
    feature_vec = np.zeros((1, cfg['vec_dim']))
    inp = np.expand_dims(inp, axis=0)
    
    if cfg['mining'] == 'offline':
        feature_vec[0, :]  = model.predict([inp, inp, inp])[0, 0, :cfg['vec_dim']]
    elif cfg['mining'] == 'online':
        feature_vec  = model.predict(inp)
        
    return feature_vec[0]

def generate_embedding(cfg, model, data):
    """ 
    Get feature vectors for a gait sequence during inferencing. Note that the function name has the word image in it because, in general terms of
    deep metrics learning usually images are the input and hence it is name as get_image_features. In this actual context it is get_sequence_features.
    
    Arguments:
    cfg - python dict, that contains the configuration. (check Deep_Metrics_Learning.ipynb)
    model - Keras.Model variable, triplet model object that has to be used for generating the feature vectors during model inference.
    data - numpy array, that contains the human joint coordinates for more than one gait sequences (whole set of data ie.., anchor set/gallery set).
    
    Returns:
    embedding_dict - python dict, that contains the feature vectors for the anchor set/gallery set of the gait sequences.
    """ 
    embedding_dict = {}
    for i in range(0, len(data)):
        embedding_dict[i] = get_image_features(cfg, model=model, inp=data[i])

    return embedding_dict

def build_annoy_index(cfg, embedding_dict={}, save_path=None):
    """ 
    Function to builds the annoy index file for the given feature vectors.
    
    Arguments:
    cfg - python dict, that contains the configuration. (check Deep_Metrics_Learning.ipynb)
    embedding_dict - python dict, that contains all the feature vectors.
    save_path - string, that contains the directory path to which the annoy index file (.ann file) has to be saved. 
    This parameter is optional, should be used only if there is a need for the index to be saved. In this project,
    there is no need to save the index file to a directory, instead it could be saved in the memory using a variable.
    
    Returns:
    t - annoy index file, that contains the indices of the feature vectors of the query/anchor sequence.
    """
    
    t = AnnoyIndex(cfg['vec_dim'], 'euclidean')

    for i in range(len(embedding_dict)):
        t.add_item(i, embedding_dict[i])
        
    t.build(100) # 50 trees
    #t.save(save_path) # save the annoy index to the disk.
    return t
    
def load_annoy_index(path_to_file, vec_dim):
    """ 
    Function to loads the annoy index file (.ann file) from the disk.
    
    Arguments:
    path_to_file - string, that has the path to the annoy index file (.ann file).
    vec_dim - Feature vector dimension.
    
    Returns:
    t - annoy index file, that contains the indices of the feature vectors of the query/anchor sequence.
    """
    t = AnnoyIndex(vec_dim, 'euclidean')
    t.load(path_to_file) 
    return t

def get_result(feature_vector, search_index, k=10):
    """ 
    Function to get the k nearest neighbors result for the given feature vector of the query/anchor sequence.
    
    Arguments:
    feature_vector - numpy array, that represents the feature vector of a gait sequence whose nearest neighbor has to be obtained.
    search_index - annoy index variable, that has the indexing of the feature vectors of the gait sequence.
    k - integer, that defines the number of nearest neighbors for the input feature vector.
    
    Returns:
    kNN[0] - python list, that contains the indices of the k-nearest neighbors of the feature vector of the query/anchor sequence.
    """
    kNN = search_index.get_nns_by_vector(feature_vector, k + 1, include_distances=True)
    return kNN[0]