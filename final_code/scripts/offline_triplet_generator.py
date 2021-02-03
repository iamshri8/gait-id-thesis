import random
import numpy as np
import copy

def offline_triplet_generator(X_train, y_train, batch_size=None, N=128):
    """
    This is a custom keras data generator function for choosing the triplets in the offline mining strategy.

    Arguments: 
    X_train - numpy array, that contains the joint coordinates of each identity in the mini-batch.
    y_train - numpy array, that contains the label for the sequences of the data in the mini-batch.
    batch_size - integer, number of sequences in the mini-batch.
    N - integer, dimensions of the feature vectors.
    
    Returns:
    It does not return anything, but yields the triplets for the mini-batch.
    """
    
    orig_index_list = np.arange(len(X_train)).tolist()
    dummy = np.zeros((1, 3 * N))

    while True:
        
        q_list = list()
        p_list = list()
        n_list = list()
        dummy_list = list()

        index_list = copy.copy(orig_index_list)

        while len(index_list) > 0:

            # Selecting the anchor/query sequence.
            index = random.choice(index_list)
            query = X_train[index]
            q_label = y_train[index]

            # Selecting the positive sequence for the selected anchor/query sequence.
            p_indices = np.where(y_train == q_label)[0]
            p_index = random.choice(np.delete(p_indices, np.where(p_indices == index)).tolist())        
            positive = X_train[p_index]

            # Selecting the negative sequence for the selected anchor/query sequence.
            n_indices = np.where(y_train != q_label)[0]
            n_index = random.choice(n_indices.tolist())
            negative = X_train[n_index]

            q_list.append(query)
            p_list.append(positive)
            n_list.append(negative)
            dummy_list.append(dummy)

            index_list.remove(index)

            if len(q_list) == batch_size or (len(index_list) == 0 and len(q_list) > 0):
                yield [np.array(q_list), np.array(p_list), np.array(n_list)], np.array(dummy_list)
                q_list = list()
                p_list = list()
                n_list = list()
                dummy_list = list()