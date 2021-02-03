import numpy as np
from sklearn.model_selection import train_test_split
import h5py, random

def read_data(path):
    """
    This function reads the HDF5 file that contains the data and extracts the poses and its labels.
    
    Arguments:
    path - string, that contains the directory path to which the hdf5 file that has the data belongs to.
    
    Returns:
    extracted_poses - numpy array, that contains the pose coordinates with actual coordinates in each frame.
    transformed_poses - numpy array, that contains the transformed/normalized (between -0.5 and 0.5) pose coordinates in each frame. 
    target - numpy array, that contains the person id and the seat id to which the sequences belong to.
    """
    with h5py.File(path, "r") as f:
        transformed_poses = np.array(f['transformed_poses'])
        extracted_poses = np.array(f['poses'])
        target = np.array(f['target'])
        
    return extracted_poses, transformed_poses, target
    
def prepare_pose(transformed_poses, seat_ids):
    """
    This function preprocess the pose coordinates so that it can be used for model training.
      
    Arguments:
    transformed_poses - numpy array, that contains the transformed/normalized (between -0.5 and 0.5) pose coordinates in each frame. 
    seat_ids - numpy array, that contains the seat ids to which each sequence belongs to. This value is discrete which indicates
    which side of the car the seat belongs to. 0 and 1 indicates that the seat is on the left side of the car, 2 and 3 indicates that 
    the seat is on the right side of the car.
    
    Returns:
    actual_poses - numpy array, that contains the pose coordinates of shape (978, 120, 22).
    """
    perspectives = np.zeros(len(transformed_poses), dtype=int)
    mask = (seat_ids == 2) | (seat_ids == 3)
    perspectives[mask] = int(1)
    print("Set 1 for seat ids 2|3 and 0 for seat ids 0|1 ::")
    print("Camera perpective shape for each sequence:: ", perspectives.shape)
    print("-----------------------------------------------------------------")

    shape = transformed_poses.shape
    actual_poses = np.zeros((shape[0], shape[2], shape[3]))

    print("Remove camera perpective dim from (978, 2, 120, 75)")
    for i in range(len(transformed_poses)):
        actual_poses[i] = transformed_poses[i, perspectives[i], :, :]

    print("Actual Pose shape after removing perspective dim:: ", actual_poses.shape)
    print("-----------------------------------------------------------------")

    print("Reshape the last dim (pose - 75) into (x,y,score - 25 x 3)")
    actual_poses = np.reshape(actual_poses, (actual_poses.shape[0], actual_poses.shape[1], actual_poses.shape[2]//3, 3))
    print("Actual Pose shape :: ", actual_poses.shape)
    print("-----------------------------------------------------------------")

    print("Eliminate score from (x,y,score) - 25 x 3 to get pose coordinates or 25-Joints(x,y) - 25 x 2")
    actual_poses = actual_poses[..., 0:2]
    print("Actual Pose shape :: ", actual_poses.shape)
    print("-----------------------------------------------------------------")

    print("Consider only 11 Joints - (7, 8, 9, 10, 11, 12, 13, 18, 20, 21, 23) out of 25 joints")
    req_joints = [7, 8, 9, 10, 11, 12, 13, 18, 20, 21, 23]

    print("Pick only the required joints from the actual pose which has all the 25 joints.")
    actual_poses = actual_poses[:, :, req_joints, :]
    print("-----------------------------------------------------------------")

    print("Reshape the joints from 2d (11 x 2) to 1d (22,)")
    actual_poses = np.reshape(actual_poses, (shape[0], shape[2], len(req_joints)*2))
    print("FINAL Actual Pose shape :: ", actual_poses.shape)
    
    return actual_poses

def get_req_ids(actual_pose, target, req_ids, person_ids):
    """
    This function provides the gait sequences and its labels for a particular set of person ids.
    
    Arguments:
    poses - numpy array, that contains the 2D human joint coordinates of the gait sequences. (shape - [978, 120, 22])
    targets - numapy array, that contains the person id and the seat id to which the gait sequences belong to.
    req_ids - python list, that contains the person ids for which the gait sequences has to be picked from the whole data.
    person_ids - numapy array, that contains only the person id of the gait sequences.
    
    Returns:
    train_x - numpy array, that contains the gait sequences of the required person ids.
    train_y - numpy array, that contains the label of the those selected gait sequences.
    """
    train_x = []
    train_y = []

    for i in req_ids:
        id_mask = (person_ids == i)
        train_x.append(actual_pose[id_mask])
        train_y.append(target[id_mask, 0])

    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)
    
    return train_x, train_y
    

def subsample(cfg, poses, targets, window_width=90, overlap=0.5):
    """
    This function subsamples the gait sequences bases on the window and the overlap parameters.
    
    Arguments:
    cfg - python dict, that contains the configuration. (check Deep_Metrics_Learning.ipynb)
    poses - numpy array, that contains the 2D human joint coordinates of the gait sequences. (shape - [978, 120, 22])
    targets - numapy array, that contains the person id and the seat id to which the gait sequences belong to.
    window_width - integer, that defines the window width of each gait sequence.
    overlap - integer, that defines the number of overlapping frames within a gait sequence.
    
    Returns:
    poses - numpy array, that contains the subsample gait sequences.
    ret_target - numpy array, that contains only the person id for each of the sequences.
    """
    joint_len = cfg['num_of_joints']
    poses = np.reshape(poses, (poses.shape[0], poses.shape[1], joint_len, 2))
    
    ret_pose = []
    ret_target = []
    
    # iterate poses
    for i in range(len(poses)):
        step = int(-window_width*overlap)  
        for ss_stop in range(poses.shape[1], 0, step): 
            if ss_stop >= window_width:
                ss = poses[i,ss_stop - window_width:ss_stop]
                ret_pose.append(ss)
                ret_target.append(targets[i])        
                
    poses = np.reshape(ret_pose, (np.array(ret_pose).shape[0], np.array(ret_pose).shape[1],
                                 joint_len * 2))

    return np.array(poses), np.array(ret_target)


def augment_data(X, y):
    """
    This function performs horizontal flip to the data and concatenates the flipped data to the original data.
    
    Arguments:
    X - numpy array, that contains the gait sequences (human joint coordinates at each frame).
    y - numpy array, that contains the label (person id) to which each sequence belongs to.
    
    Returns:
    X - numpy array, that contains the original and the horizontally flipped gait sequences.
    y - numpy array, that contains the label (person id) to which the original and the horizontally flipped sequence belongs to.    
    """
    X_flip = np.copy(X)
    X_flip[:,:,::2] = -X_flip[:,:,::2]
    X = np.concatenate((X, X_flip))
    y = np.concatenate((y, y))
        
    return X, y


def train_test(X_train, y_train, test_size=0.2, random_state=32, stratify=None):
    """
    This function splits the data into two sets (train/val set or anchor/gallery set) for training or evaluation.
    
    Arguments:
    X_train - numpy array, that contains the 2D human joint coordinates of the gait sequences.
    y_train - numpy array, that contains the label (person id) to which each sequence belongs to.    
    random_state - integer, random seed for picking the sequences for both the sets. (train/val set or anchor/gallery set)
    stratify - numpy array, based on which the class is balanced between the sequences of different identities in both the sets.
    
    Returns:
    X_train - numpy array, that contains the 2D human joint coordinates of the gait sequences. (of set 1)
    X_test - numpy array, that contains the 2D human joint coordinates of the gait sequences. (of set 2)
    y_train - numpy array, that contains the label (person id) to which each sequence belongs to in the set 1.
    y_test - numpy array, that contains the label (person id) to which each sequence belongs to in the set 2.
    """
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state, stratify=stratify)
    
    return X_train, X_test, y_train, y_test


def split_data_metrics_learning(cfg):
    """
    This function splits the original data into various subset that are required for the training and evaluation of deep metrics learning model. The original data is
    split into training set and the validation set. The split is based on the person ids, for example if there are 20 person ids from id 1, id 2, ... , id 19, id 20, then
    the training set might contain only those sequences belonging to the ids 1, 2, ... , 14, 15 and the validation set will contain those sequences from the rest of the
    person ids.
    The validation set is further split into the anchor/query set and the gallery set. Here, the split is in such a way the few of the sequences from each person id are put 
    into the anchor set and the rest goes into the gallery set. For example, if the person id 16, 17, 18, 19 and 20 belongs to the validation set, the may be 20% of the 
    sequences from each of the person ids are put into the anchor/query set and the rest goes into the gallery set.
    Once the model is trained, the feature vectors are generated for the gallery set of the validation set with the trained model. Then the feature vectors are generated for
    the sequences present in the anchor/query set and the nearest neighbors for each of them are calculated using the rank-k and the mean Average Precision evaluation metrics.
    
    Arguments:
    cfg - python dict, that contains the configuration. (check Deep_Metrics_Learning.ipynb)

    Returns:
    X_train - numpy array, that contains the 2D human joint coordinates of the gait sequences of those person ids that belong to the training set.
    X_val - numpy array, that contains the 2D human joint coordinates of the gait sequences of those person ids that belong to the validation set.
    X_train_gal - numpy array, that contains the 2D human joint coordinates of the gait sequences of those person ids that belong to the gallery set of the training set.
    X_train_anchor - numpy array, that contains the 2D human joint coordinates of the gait sequences of those person ids that belong to the anchor set of the training set.
    X_val_gal - numpy array, that contains the 2D human joint coordinates of the gait sequences of those person ids that belong to the gallery set of the validation set.
    X_val_anchor - numpy array, that contains the 2D human joint coordinates of the gait sequences of those person ids that belong to the anchor set of the validation set.
    y_train - numpy array, that contains the person id to which the gait sequences of the training set belongs to.
    y_val - numpy array, that contains the person id to which the gait sequences of the validation set belongs to.
    y_train_gal - numpy array, that contains the person id to which the gait sequences of the gallery set of the training set belongs to.
    y_train_anchor - numpy array, that contains the person id to which the gait sequences of the anchor set of the training set belongs to.
    y_val_gal - numpy array, that contains the person id to which the gait sequences of the gallery set of the validation set belongs to.
    y_val_anchor - numpy array, that contains the person id to which the gait sequences of the anchor set of the validation set belongs to.
    """
    actual_pose = cfg['actual_pose']
    target = cfg['target']
    person_ids = cfg['person_ids']
    
    # Split train and val data based on the person ids.
    all_ids = np.arange(1, 21)
    val_ids = cfg['val_ids']
    train_ids = set(all_ids).symmetric_difference(val_ids)
    
    anchor_gallery_split_size = cfg['anchor_gallery_split_size']
    window_width = cfg['window_width']
    overlap = cfg['overlap']
    random_state = cfg['random_state']
    
    # Get only the training set data and the label.
    X_train, y_train = get_req_ids(actual_pose, target, train_ids, person_ids)
    
    # Select the evaluation data that measures the performance of the model on the training set.
    train_accuracy_ids = random.sample(train_ids, len(val_ids))
    X_train_acc, y_train_acc = get_req_ids(actual_pose, target, train_accuracy_ids, person_ids)
    
    # Anchor/Gallery set split for the training set.
    X_train_gal, X_train_anchor, y_train_gal, y_train_anchor = train_test(X_train = X_train_acc, y_train = y_train_acc, 
                                                                                test_size=anchor_gallery_split_size, 
                                                                                random_state=random_state, stratify=y_train_acc)
    
    # Subsample the gait sequences of the anchor/gallery set of the training set based on the window width and the overlap.
    X_train_gal, y_train_gal = subsample(cfg, X_train_gal, y_train_gal, window_width=window_width, overlap=overlap)
    X_train_anchor, y_train_anchor = subsample(cfg, X_train_anchor, y_train_anchor, window_width=window_width, overlap=overlap)
    
    # Get only the validation set data and the label.
    X_val, y_val = get_req_ids(actual_pose, target, val_ids, person_ids)
    
    # Anchor/Gallery set split for the validation set.
    X_val_gal, X_val_anchor, y_val_gal, y_val_anchor = train_test(X_train = X_val, 
                                                                        y_train = y_val, 
                                                                        test_size=anchor_gallery_split_size, 
                                                                        random_state=random_state, 
                                                                        stratify=y_val)
    
    
    # If data augmentation parameter is set to True in the configuration dictionary, data augmentation is done for the training set.
    if cfg['augment_data']:
        X_train, y_train = augment_data(X_train, y_train)
    
    # Subsample the gait sequences of the whole training set based on the window width and the overlap.
    X_train, y_train = subsample(cfg, X_train, y_train, window_width=window_width, overlap=overlap)
    
    # Subsample the gait sequences of the anchor/gallery set of the validation set based on the window width and the overlap.
    X_val_gal, y_val_gal = subsample(cfg, X_val_gal, y_val_gal, window_width=window_width, overlap=overlap)
    X_val_anchor, y_val_anchor = subsample(cfg, X_val_anchor, y_val_anchor, window_width=window_width, overlap=overlap)
    
    # Concatenate the gallery and anchor set of the validation data and label as a whole. This is just to maintain the train-val uniformity and 
    # is not used anywhere in the project.
    X_val, y_val = np.concatenate((X_val_gal, X_val_anchor)), np.concatenate((y_val_gal, y_val_anchor))
    
    return X_train, X_val, X_train_gal, X_train_anchor, X_val_gal, X_val_anchor, y_train, y_val, y_train_gal, y_train_anchor, y_val_gal, y_val_anchor