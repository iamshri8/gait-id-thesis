import tensorflow as tf
import keras
from keras.callbacks import Callback
import numpy as np
from annoy import AnnoyIndex
from collections import defaultdict
import os
from .evaluate import get_image_features, get_result, generate_embedding, build_annoy_index
from .metrics import get_metrics

class EvalOnEpochEnd(Callback):
    """
    Custom callback class that evaluates the model after every n epochs (where n epochs mean, the frequency at which the the model has to be evaluated.).
    """
    def __init__(self, cfg, train_gallery=(), train_anchor=(), val_gallery=(), val_anchor=(), interval=1):
        """
        Constructor that initializes input required for model evaluation after every n epochs.

        Arguments: 
        cfg - python dict, that contains the configuration. (check Deep_Metrics_Learning.ipynb)    
        train_gallery - python tuple, that contains the data and the label of the gallery set from the training set.
        train_anchor - python tuple, that contains the data and the label of the anchor set from the training set.
        val_gallery - python tuple, that contains the data and the label of the gallery set from the validation set.
        val_anchor - python tuple, that contains the data and the label of the anchor set from the training set.
        interval - integer, that specifies how frequent/often(once in every n epochs where n is the interval) model evaluation should take place.

        Returns:
        None - No return, only initialization.

        """
        super(Callback, self).__init__()

        self.cfg = cfg
        
        self.X_train_gal, self.y_train_gal = train_gallery
        self.X_train_anc, self.y_train_anc = train_anchor
        
        self.X_val_gal, self.y_val_gal = val_gallery
        self.X_val_anc, self.y_val_anc = val_anchor
        
        # lists that contain rank-1 accuracy and mean Average Precision of the model on the validation set.
        self.val_rank1_list = list()
        self.val_mAP_list = list()
        
        # lists that contain rank-1 accuracy and mean Average Precision of the model on the training set.
        self.train_rank1_list = list()
        self.train_mAP_list = list()
        
        # this variable stores the best rank-1 accuracy on the training set.
        self.max_train_acc = 0.0
        
        # this variable saves the model that has the best training rank-1 accuracy to the directory path. 
        self.train_model_path = ""
        
        # this variable stores the best rank-1 accuracy on the validation set.
        self.max_val_acc = 0.0
        
        # this variable saves the model that has the best validation rank-1 accuracy to the directory path. 
        self.val_model_path = ""

    def on_epoch_end(self, epoch, logs={}):
        
        """
        Overriding the method on_epoch_end() that is called at the end of every epoch during model training. For example, if the interval is set to 3, then
        this method is called once in 3 epochs.

        Arguments: 
        epoch - integer, that indicates current epoch.
        logs - python dict, that contain values of the default metics like training loss and validation loss at the current epoch.

        Returns:
        None - this method doesn't return anything, it only evaluates the model once every n epochs.
        """
            
        # Evaluation of the model on the training set.
        embed_dict = generate_embedding(self.cfg, self.model, self.X_train_gal)
        annoy_index = build_annoy_index(self.cfg, embed_dict)
        train_rank1_acc, train_mAP = get_metrics(self.cfg, self.model, 
                                                 self.X_train_anc, self.y_train_anc, 
                                                 self.X_train_gal, self.y_train_gal, 
                                                 annoy_index, self.cfg['vec_dim'])
        print("\t Train Metrics - epoch: {:d} - rank1_acc: {:.6f} - mAP: {:.6f}".format(epoch+1, train_rank1_acc, train_mAP))
        
        self.train_rank1_list.append(train_rank1_acc)
        self.train_mAP_list.append(train_mAP)
        
        if train_rank1_acc > self.max_train_acc:
            self.max_train_acc = train_rank1_acc
            val_ids = str(self.cfg['val_ids'])
            window_width = str(self.cfg['window_width'])
            vec_dim_str = str(self.cfg['vec_dim'])
            mining = self.cfg['mining']
            train_model_path = os.path.join(self.cfg['ckpt_dir'],
                                    f'model-train-{val_ids}-{window_width}-{vec_dim_str}-{mining}.h5')
            self.model.save(train_model_path)
            self.train_model_path = train_model_path
        
        # Evaluation of the model on the validation set.
        embed_dict = generate_embedding(self.cfg, self.model, self.X_val_gal)
        annoy_index = build_annoy_index(self.cfg, embed_dict)
        val_rank1_acc, val_mAP = get_metrics(self.cfg, self.model, 
                                             self.X_val_anc, self.y_val_anc, 
                                             self.X_val_gal, self.y_val_gal, 
                                             annoy_index, self.cfg['vec_dim'])
        print("\t Val Metrics - epoch: {:d} - rank1_acc: {:.6f} - mAP: {:.6f}".format(epoch+1, val_rank1_acc, val_mAP))
        
        self.val_rank1_list.append(val_rank1_acc)
        self.val_mAP_list.append(val_mAP)

        if val_rank1_acc > self.max_val_acc:
            self.max_val_acc = val_rank1_acc
            val_ids = str(self.cfg['val_ids'])
            window_width = str(self.cfg['window_width'])
            vec_dim_str = str(self.cfg['vec_dim'])
            mining = self.cfg['mining']
            val_model_path = os.path.join(self.cfg['ckpt_dir'],
                                    f'model-val-{val_ids}-{window_width}-{vec_dim_str}-{mining}.h5')
            self.model.save(val_model_path)
            self.val_model_path = val_model_path
            
    def get_all_metrics(self):
        """
        This method returns all the evaluation metrics recorded during training.
        
        Arguments:
        self - default object that maintains the state of all the variable of the class and is accessible anywhere within the class.
        
        Returns:
        lists that contain the rank-1 accuracy and the mean Average Precision of the model evaluated on the training set as well the vaidation set.
        """
        return self.train_rank1_list, self.train_mAP_list, self.val_rank1_list, self.val_mAP_list
    
    def get_best_train_model(self):
        """
        This method returns the path to the directory at which the model with best rank-1 accuracy on the training set is saved.
        
        Arguments:
        self - default object that maintains the state of all the variable of the class and is accessible anywhere within the class.
        
        Returns:
        Path to the directory at which the model with best rank-1 accuracy on the training set is saved.
        """        
        return self.train_model_path
    
    def get_best_val_model(self):
        """
        This method returns the path to the directory at which the model with best rank-1 accuracy on the validation set is saved.
        
        Arguments:
        self - default object that maintains the state of all the variable of the class and is accessible anywhere within the class.
        
        Returns:
        Path to the directory at which the model with best rank-1 accuracy on the validation set is saved.
        """
        return self.val_model_path
    
