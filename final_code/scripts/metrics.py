import numpy as np
from annoy import AnnoyIndex
from collections import defaultdict
from .evaluate import get_image_features, get_result

def mean_average_precision(rs):
   """Score is mean average precision
   Relevance is binary (nonzero is relevant).
   Args:
       rs: Iterator of relevance scores (list or numpy) in rank order
           (first element is the first item)
   Returns:
       Mean average precision
   """
   return np.mean([average_precision(r) for r in rs])


def average_precision(r):
   """Score is average precision (area under PR curve)
   Relevance is binary (nonzero is relevant).
   Args:
       r: Relevance scores (list or numpy) in rank order
           (first element is the first item)
   Returns:
       Average precision
   """
   r = np.asarray(r) != 0
   out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
   if not out:
       return 0.
   return np.mean(out)



def precision_at_k(r, k):
   """Score is precision @ k
   Relevance is binary (nonzero is relevant).

   Traceback (most recent call last):
       File "<stdin>", line 1, in ?
   ValueError: Relevance score length < k
   Args:
       r: Relevance scores (list or numpy) in rank order
           (first element is the first item)
   Returns:
       Precision @ k
   Raises:
       ValueError: len(r) must be >= k
   """
   assert k >= 1
   r = np.asarray(r)[:k] != 0
   if r.size != k:
       raise ValueError('Relevance score length < k')
   return np.mean(r)

def get_metrics(cfg, model, X_anchor, y_anchor, X_gal, y_gal, annoy_index, vec_dim):
    """
    Print all evaluation metrics, rank-1, rank-5, rank-10 accuracy, mean Average Precision, most voted rank-10 accuracy and average correct resutls(accuracy).
    
    Arguments:
    cfg - python dict, that contains the configuration. (check Deep_Metrics_Learning.ipynb)    
    model - Keras.Model variable, that contains the model object for which the metrics should be evaluted.
    X_anchor - numpy array, that contains the anchor set data.
    y_anchor - numpy array, that contains the anchor set labels.
    X_gal - numpy array, that contains the gallery set data.
    y_gal - numpy array, that contains the gallery set labels.
    annoy_index - annoy index, that has the indexing for the gallery set feature vectors.
    vec_dim - integer, that denotes the dimensions of the feature vectors representing a gait sequence.
    
    Returns:
    rank-1 accuracy - floating point number, rank-1 accuracy.
    mean Average Precision - floating point number, mean Average Precision.
    """
    rank10_acc = 0
    rank5_acc = 0
    rank1_acc = 0
    avg_acc = 0
    vote_res = 0

    l2 = []
    for anchor in range(0, len(X_anchor)):
        res = get_result(get_image_features(cfg, model, X_anchor[anchor]), annoy_index)
        vote = defaultdict(int)
        # Accuracy
        correct = 0
        for i in res[:10]:
            vote[y_gal[i]] += 1

        max_key = max(vote, key=vote.get)
        if max_key == y_anchor[anchor]:
            vote_res += 1
        

        for recomm in res[:10]:
            if y_gal[recomm] == y_anchor[anchor]:
                correct += 1     

        avg_acc += correct/len(res)

        # Mean Average Precision
        l1 = []
        for recomm in res[:10]:
            if y_gal[recomm] == y_anchor[anchor]:
                correct += 1
                l1.append(1)
            else:
                l1.append(0)
        l2.append(l1) 

        # Rank10 Accuracy
        for each_val in res[:10]:
            if y_gal[each_val] == y_anchor[anchor]:
                rank10_acc += 1
                break
        
        # Rank5 Accuracy
        for each_val in res[:5]:
            if y_gal[each_val] == y_anchor[anchor]:
                rank5_acc += 1
                break

        # Rank1 Accuracy
        for each_val in res[:1]:
            if y_gal[each_val] == y_anchor[anchor]:
                rank1_acc += 1
                break

    print("Avg acc is :: {avg_acc}".format(avg_acc = avg_acc/len(X_anchor)))
    print("Rank 10 acc is :: {rank10_acc}".format(rank10_acc = rank10_acc/len(X_anchor)))
    print("Rank 5 acc is :: {rank5_acc}".format(rank5_acc = rank5_acc/len(X_anchor)))
    print("Rank 1 acc is :: {rank1_acc}".format(rank1_acc = rank1_acc/len(X_anchor)))
    print("Mean Avg Precision is :: {mAP}".format(mAP=mean_average_precision(l2)))
    print("Vote res :: ", vote_res/len(X_anchor))

    return rank1_acc/len(X_anchor), mean_average_precision(l2)