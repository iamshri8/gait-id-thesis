import tensorflow as tf
import keras
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
import mlflow
import mlflow.keras
import numpy as np

from scripts.data import split_data_metrics_learning
from .model_config import base_model, offline_triplet_network, online_triplet_network
from .losses import offline_triplet_loss
from tensorflow_addons.losses.triplet import triplet_semihard_loss
from .offline_triplet_generator import offline_triplet_generator
from .evaluate import generate_embedding, build_annoy_index
from .metrics import get_metrics
from .model_eval_on_epoch_end import EvalOnEpochEnd
from .mlflow_logger import MLFlowLogger

def train(cfg):
    """
    Function for training the deep metrics learning models.

    Arguments: 
    cfg - python dict, that contains the configuration. (check Deep_Metrics_Learning.ipynb)    
    
    Returns:
    None - this method doesn't return anything, it only trains the model for the data provided for training.
    """

    # Initialize mlflow logger object.
    mlflow_logger = MLFlowLogger(cfg)
    
    # Split the training set and the validation set for model training.
  
    # Initialize keras custom callback function for evaluating the model during traininig.
    callbacks = EvalOnEpochEnd(cfg=cfg, 
                                   train_gallery=(X_train_gal, y_train_gal), 
                                   train_anchor=(X_train_anchor, y_train_anchor),
                                   val_gallery=(X_val_gal, y_val_gal), 
                                   val_anchor=(X_val_anchor, y_val_anchor),
                                   interval=cfg['eval_interval'])

    # Initialize run name for mlflow logger.
    run_name = str(cfg['val_ids'])+"-"+str(cfg['window_width'])+"-"+str(cfg['vec_dim'])+"-"+cfg['mining']
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_uuid

        mlflow_logger.log_params()
        mlflow_logger.log_tags(run_id)

        # This if statement is applicable when the mining strategy for training triplet network is "Offline".
        if cfg['mining'] == 'offline':
            
            # Configure the model for offline mining. 
            base_model_obj = base_model(cfg)
            model = offline_triplet_network(cfg, base_model_obj)
            opt = Adam(learning_rate=cfg['learning_rate'])
            model.compile(optimizer=opt, loss=offline_triplet_loss)

            # Custom data generator function is need only for the offline mining strategy.
            # Initialize a custom keras data generator for selecting triplets in the training set. (offline mining)
            train_gen = offline_triplet_generator(X_train, y_train, cfg['batch_size'], cfg['vec_dim'])
            
            # Initialize a custom keras data generator for selecting triplets in the validation set. (offline mining)
            val_gen = offline_triplet_generator(X_val, y_val, cfg['batch_size'], cfg['vec_dim'])

            train_history = model.fit(train_gen,
                                batch_size=cfg['batch_size'],
                                epochs=cfg['epochs'],
                                steps_per_epoch=len(X_train) // cfg['batch_size'],
                                validation_data=val_gen,
                                validation_steps = len(X_val) // cfg['batch_size'],
                                callbacks = [callbacks],
                                verbose=1) 
            
            mlflow_logger.log_train_metrics(train_history, callbacks)
            model = tf.keras.models.load_model(filepath=callbacks.get_best_val_model(), custom_objects={'offline_triplet_loss': offline_triplet_loss})
            mlflow_logger.log_model(model) 
            del model
            
        # This elif statement is applicable when the mining strategy for training triplet network is "Online".
        elif cfg['mining'] == 'online':
            
            # Configure the model for online mining. 
            model = online_triplet_network(cfg)
            opt = Adam(learning_rate=cfg['learning_rate'])
            model.compile(optimizer=opt, loss=triplet_semihard_loss)

            train_history = model.fit(x=X_train,
                                y=y_train,
                                batch_size=cfg['batch_size'],
                                epochs=cfg['epochs'],
                                callbacks=[callbacks],
                                verbose=1,
                                validation_data=(X_val, y_val),
                                shuffle=True,
                                max_queue_size=10,
                                workers=8,
                                use_multiprocessing=True)

            mlflow_logger.log_train_metrics(train_history, callbacks)
            model = tf.keras.models.load_model(filepath=callbacks.get_best_val_model(), custom_objects={'triplet_semihard_loss': triplet_semihard_loss})
            mlflow_logger.log_model(model) 
            del model